import cv2
import numpy as np

from imageProcessor import ImageProcessor
from sudoku import Sudoku


def main():
    cap = cv2.VideoCapture(0)
    processor = ImageProcessor()

    reset = True
    solution = None
    valid_frame_count = 0
    digit_images = dict()
    digit_flags_history = list()
    window_name = "AR Sudoku Solver"

    while(cap.isOpened()):
        frame = cap.read()[1]
        rectangles = processor.extract_rectangles(frame)

        if rectangles:
            valid_sudoku = False
            for rectangle in rectangles:
                rot_matrix, pt_frame = processor.perspective_transform(rectangle, frame)
                cv2.imshow("sudoku", pt_frame)
                sudoku_cells = processor.extract_cells(pt_frame)
                digit_flags = list()

                for index, cell in enumerate(sudoku_cells):
                    cell = cell.reshape(28,28)
                    flag, digit_img = processor.crop_digit(cell)
                    digit_flags.append(flag)
                    if flag:
                        digit_images.setdefault(index, []).append(digit_img.flatten())
                
                if sum(digit_flags) > 16: # a valid (unique) sudoku will have atleast 17 digits
                    valid_sudoku = True
                    if not reset and solution is not False and solution is not None:
                        frame = processor.display_solution(pt_frame, solution, 
                                                                 digits_flag_merged, frame, rot_matrix)
                        cv2.imshow(window_name, frame)
                        digit_images.clear()
                    elif not reset:
                        reset = True
                        solution = None
                        digit_images.clear()
                        valid_frame_count = 0
                    break

            if not valid_sudoku:
                reset = True
                solution = None
                digit_flags_history.clear()
                digit_images.clear()
                display_msg = "Please bring the sudoku closer"
                frame = cv2.putText(frame, display_msg, (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX , 1, 
                                    (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, frame)

            elif reset:
                valid_frame_count += 1
                digit_flags_history.append(digit_flags)
                if valid_frame_count >= 7:
                    digits_flag_merged = np.sum(np.array(digit_flags_history),axis=0).astype(np.bool)
                    digit_images_processed = list()
                    try:
                        for cell_index in range(81):
                            if digits_flag_merged[cell_index]:
                                digit_images_processed.append(np.mean(digit_images[cell_index],axis=0))
                    except:
                        cv2.imshow(window_name, frame)
                        continue
                    reset = False
                    digits = processor.digit_recogniser(digit_images_processed)
                    print("Identified digits : ", digits)
                    board = np.zeros(81)
                    board[digits_flag_merged] = digits
                    sudoku = Sudoku(''.join(map(str, board.astype(np.int))))
                    solution = sudoku.solve()
                    print("Solution : ", solution)
                    if solution is not False:
                        frame = processor.display_solution(pt_frame, solution, 
                                                           digits_flag_merged, frame, rot_matrix)
                cv2.imshow(window_name, frame)

        else:
            reset = True
            solution = None
            valid_frame_count = 0
            digit_images.clear()
            digit_flags_history.clear()
            cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Program has been stopped by the user!")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()