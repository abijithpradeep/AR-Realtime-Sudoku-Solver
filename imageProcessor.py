import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import os
tf.disable_v2_behavior()


class ImageProcessor:
    def __init__(self):
        self.x = None
        self.y = None
        self.prob =  None
        self.pred_label = None
        self.session = None
        self.load_model()

    def preprocess(self, image):
        """
        Preprocess the given image by converting to grayscale,
        applying gaussian blur, morphological transformation and
        adaptive threshold.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (7, 7), 0)  
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) 
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY, 19, 2)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((2, 2)))

        return image

        
    def extract_rectangles(self, image):
        """
        Extract all the rectangles in the image.
        """
        image = self.preprocess(image)
        image_area = image.shape[0] * image.shape[1]
        threshold_area = image_area/8.5
        contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        rectangles = list()
        for contour in contours:
            polygon = cv2.approxPolyDP(contour,15, True)
            if len(polygon) == 4 and cv2.isContourConvex(polygon):  
                rectangle = polygon
                if cv2.contourArea(rectangle) > threshold_area:
                    rectangles.append(rectangle)

        return rectangles

    
    def perspective_transform(self, rectangle, image):
        """
        Peform perspective transform on the extracted sudoku.
        """
        width, height = 252, 252
        topleft = min(rectangle, key=lambda point: point[0,0] + point[0,1])
        bottomright = max(rectangle, key=lambda point: point[0,0] + point[0,1])
        topright = max(rectangle, key=lambda point: point[0,0] - point[0,1])
        bottomleft = min(rectangle, key=lambda point: point[0,0] - point[0,1])
        source = np.array([topleft, topright, bottomleft, bottomright], dtype = np.float32)
        destination = np.array([
                                 [0, 0],
                                 [width, 0],
                                 [0, height], 
                                 [width, height]
                               ],
                                 dtype=np.float32)
        rot_matrix = cv2.getPerspectiveTransform(source, destination)
        transformmed_image = cv2.warpPerspective(image, rot_matrix, (width, height))

        return rot_matrix, transformmed_image


    def extract_cells(self, sudoku_img):
        """
        Cropping and extracting the individual cells from the sudoku board.
        """
        bi_sudoku = cv2.cvtColor(sudoku_img, cv2.COLOR_BGR2GRAY)
        sudoku_img = cv2.adaptiveThreshold(bi_sudoku, 255, 
                                          cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 6)
        sudoku_img = cv2.morphologyEx(sudoku_img, cv2.MORPH_OPEN,
                                      np.ones((2, 2), np.uint8))
        cells = list()
        for row in np.split(sudoku_img, 9, axis=0):
            for cell in np.split(row, 9, axis=1):
                cells.append(cell.flatten())

        return cells
        

    def crop_digit(self, img):
        """
        Extracts and crops the digit image from the cell image.
        """
        digit_size = (16, 20)
        contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        boundaries = list()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 14 > x > 3 and 14 > y > 1 \
                and 20 > h > 7 and 20 > w > 3 \
                and 300 > w * h > 21 and h > w :
                boundaries.append((x, y, w, h))

        if len(boundaries):
            x, y, w, h = max(boundaries, key=lambda c: c[2] * c[3])
            cropped_digit_img = img[y : (y + h) + 1, 
                                    x : (x + w) + 1]
            if w * h <= x * y:
                cropped_digit_img = cv2.resize(cropped_digit_img, digit_size,
                                               interpolation=cv2.INTER_LINEAR)
            else:
                cropped_digit_img = cv2.resize(cropped_digit_img, digit_size,
                                               interpolation=cv2.INTER_AREA)
            digit_img = np.zeros((28, 28))
            margin = (np.round((28 - digit_size[0])//2), np.round((28 - digit_size[1])//2))
            digit_img[margin[1] : margin[1] + digit_size[1], 
                      margin[0] : margin[0] + digit_size[0]] = cropped_digit_img

            return True, digit_img
        else:
            return False, np.zeros((28, 28))


    def load_model(self):
        """
        Loads the pre-trained digit classifier model
        """
        model_path = './model/digit_classifier.meta'
        if not os.path.exists(model_path):
            print ("Digit recognisation model not found ! \nExiting the program")
            exit()
        self.session = tf.Session()
        saver = tf.train.import_meta_graph(model_path)
        saver.restore(self.session, tf.train.latest_checkpoint('./model/'))
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name("input_image:0")
        self.y = graph.get_tensor_by_name("input_label:0")
        self.prob = graph.get_tensor_by_name("keep_prob:0")
        self.pred_label = graph.get_tensor_by_name("pred_label:0")


    def digit_recogniser(self, dig_imgs):
        """
        Recognise the digits.
        """
        prediction = self.pred_label.eval(
                                          session=self.session, 
                                          feed_dict={self.x:dig_imgs, 
                                                    self.y:np.zeros((81,9)), 
                                                    self.prob:1.0}
                                         )
            
        return prediction + 1


    def display_solution(self, img, solution, digit_flags, frame, rot_matrix):
        """
        Writes the solution onto the actual frame.
        """
        sudoku_width, sudoku_height = img.shape[:2][::-1]
        cell_width  = sudoku_width / 9
        cell_height = sudoku_height / 9

        for cell_index in range(81):
            if not digit_flags[cell_index]:
                output_cell_x = int((cell_index % 9) * cell_width) + 7
                output_cell_y = int((cell_index // 9) * cell_height) + 17
                cv2.putText(
                             img, solution[cell_index], 
                             (output_cell_x, output_cell_y), 
                             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                             0.55, (255, 10, 10), 1, cv2.LINE_AA
                           )
        persp = cv2.warpPerspective(
                                      img, rot_matrix, 
                                      frame.shape[:2][::-1], 
                                      frame.copy(), 
                                      cv2.WARP_INVERSE_MAP
                                    )
        sudoku_portion = frame*(persp == 0)
        cv2.imwrite("output/solvedSudoku.jpg", img)
        bg = persp*(persp!= 0) 

        return sudoku_portion + bg
