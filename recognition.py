import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import time
import os
import requests
from PIL import Image
import cv2
import mysql.connector
from tensorflow.keras.models import load_model

from cropper_validations import find_outermost_rectangle, find_fourth_point_v2, extrapolate_for_direction_right, \
    get_markers_direction, extrapolate_for_direction_bottom, extrapolate_for_direction_left, \
    extrapolate_for_direction_top, is_skewed, dimension_checker
from fix_tilted_image import correct_image_tilt
from save_images import crop_validation, crop_image, response, new_session_dir, image_download_domain


# Train the CNN model on MNIST dataset
# def train_mnist_cnn_model():
#     start_total = time.time()
#
#     # Load the MNIST dataset
#     (trainX, trainY), (testX, testY) = mnist.load_data()
#
#     # Reshape the data to fit the model (28x28 images with 1 color channel)
#     trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
#     testX = testX.reshape((testX.shape[0], 28, 28, 1))
#
#     # Normalize pixel values to be between 0 and 1
#     trainX = trainX.astype('float32') / 255.0
#     testX = testX.astype('float32') / 255.0
#
#     # One-hot encode the labels
#     trainY = to_categorical(trainY, 10)
#     testY = to_categorical(testY, 10)
#
#     # Build the CNN model
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(10, activation='softmax'))
#
#     # Compile the model
#     model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
#
#     # Train the model
#     start_time = time.time()
#     history = model.fit(trainX, trainY, epochs=5, batch_size=128, validation_data=(testX, testY))
#     end_time = time.time()
#     duration = end_time - start_time
#     print(f"Training duration: {duration:.2f} seconds")
#
#     # Evaluate the model on the test set
#     test_loss, test_acc = model.evaluate(testX, testY)
#     print(f"Test accuracy: {test_acc}")
#
#     # Save the model
#     model.save('mnist_cnn_model.h5')
#
#     # Plot training & validation accuracy values
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('Model accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.show()
#
#     # Plot training & validation loss values
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.show()
#
#     return model


# Function to start execution
def startExecution(response, new_session_dir, image_download_domain):
    print("\n" * 2)
    print("startExecution ...", response)

    s3_path = response.get("s3_path")
    file_name = response.get("image_name")
    new_session_image_store = new_session_dir

    if not os.path.exists(new_session_image_store):
        os.makedirs(new_session_image_store, exist_ok=True)

    image_url = f"{image_download_domain}/{s3_path}"
    fetch_response = requests.get(image_url)
    image_file_path = os.path.join(new_session_image_store, file_name)
    print(image_file_path)

    if fetch_response.status_code == 200:
        print(fetch_response)
        with open(image_file_path, "wb") as f:
            print("open")
            f.write(fetch_response.content)
            print("image downloaded successfully ")

        initial_cropped_image_path = "/home/madhu/cropped_images/20240621022617_pdf_page_3.png"
        crop_image_based_on_markers_yoga(image_file_path, initial_cropped_image_path)

        crop_image_based_on_markers_yoga(image_file_path, initial_cropped_image_path)
        formatted_file_name = file_name.replace('.png', '', 1)
        secondary_cropped_image_path = os.path.join(os.path.dirname(new_session_dir),f"{formatted_file_name}_marks.png")
        crop_image_based_on_markers_yoga(initial_cropped_image_path, secondary_cropped_image_path)
        # # recognize_marks(secondary_cropped_image_path)
    else:
        print(f"Failed to download the image. Status code: {fetch_response.status_code}")

response = {
        "s3_path": "testing/part-c/20240621022617/20240621022617_pdf_page_1719309550491.png",
        "image_name": "20240621022617_pdf_page_3.png"
    }
new_session_dir = "/home/madhu/cropped_images"
image_download_domain = "https://docs.exampaper.vidh.ai"


# Function to get s3_path from the database
def get_s3_path_from_database(image_name):
    try:
        conn = mysql.connector.connect(
            host="34.131.182.12",
            user="rootuser",
            password="1234",
            database="vidhai_ms_solutions_dev"
        )
        cursor = conn.cursor()
        sql = "SELECT s3_path FROM ocr_scanned_part_c_v1 WHERE image_name = %s"
        image_name = '20240621022617_pdf_page_3.png'
        cursor.execute(sql, (image_name,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result:
            return result[0]
        else:
            return None
    except Exception as e:
        print(f"Error fetching s3_path from database: {e}")
        return None


# Function to crop image based on markers
def crop_image_based_on_markers_yoga(image_path, output_path):
    try:
        is_titlted = correct_image_tilt(image_path, image_path)
        global count
        error_code = None
        error_reason = None
        det_marker_cnt = None
        print("try with outer markers...")

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image.")
            return

        # Make a copy of the original image to draw markers on
        marked_image = image.copy()

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply localized thresholding to get a binary image
        block_size = 153  # Larger block size for better local adaptation
        constant = 20  # Adjust constant for better thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size,
                                       constant)

        # imS = cv2.resize(thresh, (960, 540))
        # cv2.imshow("thresh", imS)
        # cv2.waitKey()
        # cv2.destroyAllWindows
        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours based on area (descending order)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Draw all contours on the marked image
        # cv2.drawContours(marked_image, contours, -1, (0, 0, 255), 2)  # Draw all contours in red
        # imS = cv2.resize(marked_image, (960, 540))
        # cv2.imshow("thresh", imS)
        # cv2.waitKey()
        # cv2.destroyAllWindows
        # List to hold the marker points
        markers = []

        expected_marker_size = (52, 53)
        min_area = 2300
        max_area = 3300

        print(min_area, max_area)
        # Iterate over the contours to find the largest square markers
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)  # Increased accuracy
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # print("area = ", area)
            # print(x,y,w,h)
            # print(aspect_ratio)

            if min_area <= area <= max_area and 0.7 <= aspect_ratio <= 1.2 and 50 <= w < 62 and 50 <= h < 62:
                ## suj check

                # print("area = ", area)
                # print(x,y,w,h)
                # print(aspect_ratio)
                # print("true area: ", area)
                marker = (x + w // 2, y + h // 2)
                markers.append(marker)
                # cv2.circle(marked_image, marker, 10, (0, 255, 0), -1)

                # if len(markers) == 4:
                #     break

        print("Detected markers:", markers)
        det_marker_cnt = len(markers)
        ref_markers = markers

        if len(markers) > 4:
            print("found marker is greater than 4")
            is_titlted = correct_image_tilt(image_path, image_path)
            print("is_titlted = ", is_titlted)
            print("count = ", count)
            # if is_titlted and count < 1:
            #     count += 1
            #     st,error_code, error_reason, det_marker_cnt = crop_image_based_on_markers_v6(image_path, output_path)
            #     return st, error_code, error_reason, det_marker_cnt
            outermost_rectangle = find_outermost_rectangle(markers)
            print("outermost_rectangle = ", outermost_rectangle)
            markers = outermost_rectangle


        elif len(markers) == 4:
            print("found markers is 4")
            markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x

            # Manually assign top-left, top-right, bottom-left, bottom-right
            top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
            bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
            markers = [top_left, top_right, bottom_left, bottom_right]
            print("sorted marker = ", markers)


        elif len(markers) == 3:
            print("only 3 markers is being found")
            fourth_point = find_fourth_point_v2(markers)
            if fourth_point:
                markers.append(fourth_point)
                # markers = find_fourth_point_v2(markers)
                markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
                print("sorted = ", markers)
                # Manually assign top-left, top-right, bottom-left, bottom-right
                top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
                bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
                markers = [top_left, top_right, bottom_left, bottom_right]
                print("sorted marker = ", markers)
            else:
                print("4th point not found..")
                error_code = 103
                error_reason = "fourth point prediction failed"

        elif len(markers) == 2:

            print("tilted good")
            print("is_titlted = ", is_titlted)
            print("count = ", count)
            # if is_titlted and count < 1:
            #     count += 1
            #     st,error_code, error_reason, det_marker_cnt = crop_image_based_on_markers_v6(image_path, output_path)

            with Image.open(image_path) as img:
                image_w, image_h = img.size

            print(image_w, image_h)
            direction = get_markers_direction(image_w, image_h, markers)
            print("direction:", direction)
            if not direction:
                print("Invalid direction!!!")

            elif direction and direction == "right":
                markers = extrapolate_for_direction_right(markers)
                print(markers)
                if len(markers) == 4:
                    markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
                    print("sorted = ", markers)
                    # Manually assign top-left, top-right, bottom-left, bottom-right
                    top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
                    bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
                    markers = [top_left, top_right, bottom_left, bottom_right]
                    print("sorted marker = ", markers)
                else:
                    print("marker count mismatch...")
                    error_code = 103
                    error_reason = "failed right"

            elif direction and direction == "bottom":
                print("bottom")
                markers = extrapolate_for_direction_bottom(markers)
                print(markers)
                if len(markers) == 4:
                    markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
                    print("sorted = ", markers)
                    # Manually assign top-left, top-right, bottom-left, bottom-right
                    top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
                    bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
                    markers = [top_left, top_right, bottom_left, bottom_right]
                    print("sorted marker = ", markers)
                else:
                    print("marker count mismatch...")
                    error_code = 103
                    error_reason = "failed bottom"

            elif direction and direction == "left":
                print("left")
                markers = extrapolate_for_direction_left(markers)
                print(markers)
                if markers and len(markers) == 4:
                    markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
                    print("sorted = ", markers)
                    # Manually assign top-left, top-right, bottom-left, bottom-right
                    top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
                    bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
                    markers = [top_left, top_right, bottom_left, bottom_right]
                    print("sorted marker = ", markers)
                else:
                    print("marker count mismatch...")
                    error_code = 103
                    error_reason = "left"
                    return False, error_code, error_reason, det_marker_cnt
            # elif direction and direction == "diag-back":
            #     print("diag-back")
            #     markers = extrapolate_for_direction_diag_back (markers)
            #     print (markers)
            #     if len(markers) == 4:
            #         markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
            #         print("sorted = ", markers)
            #         # Manually assign top-left, top-right, bottom-left, bottom-right
            #         top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
            #         bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
            #         markers = [top_left, top_right, bottom_left, bottom_right]
            #         print("sorted marker = ", markers)
            #     else:
            #         print("marker count mismatch...")
            #         error_code = 103
            #         error_reason = "two point prediction failed"

            elif direction and direction == "top":
                print("top")
                markers = extrapolate_for_direction_top(markers)
                print(markers)
                if len(markers) == 4:
                    markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
                    print("sorted = ", markers)
                    # Manually assign top-left, top-right, bottom-left, bottom-right
                    top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
                    bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
                    markers = [top_left, top_right, bottom_left, bottom_right]
                    print("sorted marker = ", markers)
                else:
                    print("marker count mismatch...")
                    error_code = 103
                    error_reason = "failed top"

            elif direction and direction == "diag-back":
                print("diag-back")
                markers = extrapolate_for_direction_top(markers)
                print(markers)
                if len(markers) == 4:
                    markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
                    print("sorted = ", markers)
                    # Manually assign top-left, top-right, bottom-left, bottom-right
                    top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
                    bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
                    markers = [top_left, top_right, bottom_left, bottom_right]
                    print("sorted marker = ", markers)
                else:
                    print("marker count mismatch...")
                    error_code = 103
                    error_reason = "failed diag-back"

            elif direction and direction == "diag-front":
                print("diag-back")
                markers =  markers = extrapolate_for_direction_top(markers)
                print(markers)
                if len(markers) == 4:
                    markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
                    print("sorted = ", markers)
                    # Manually assign top-left, top-right, bottom-left, bottom-right
                    top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
                    bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
                    markers = [top_left, top_right, bottom_left, bottom_right]
                    print("sorted marker = ", markers)
                else:
                    print("marker count mismatch...")
                    error_code = 103
                    error_reason = "failed diag-front"

            # elif direction and direction == "diag-front":
            #     print("diag-back")
            #     markers = extrapolate_for_direction_diag_front (markers)
            #     print (markers)
            #     if len(markers) == 4:
            #         markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
            #         print("sorted = ", markers)
            #         # Manually assign top-left, top-right, bottom-left, bottom-right
            #         top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
            #         bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
            #         markers = [top_left, top_right, bottom_left, bottom_right]
            #         print("sorted marker = ", markers)
            #     else:
            #         print("marker count mismatch...")
            #         error_code = 103
            #         error_reason = "two point prediction failed"

            else:
                error_code = 103
                error_reason = direction
                return False, error_code, error_reason, det_marker_cnt

        else:
            print("crop failed...")
            print(f"no of markers detected is {len(markers)}")
            print("Trying with tilt correction...")
            # marking(marked_image, ref_markers)
            # is_titlted = correct_image_tilt(image_path, image_path)
            # print("is_titlted = ", is_titlted)
            # print("count = ", count)
            # if is_titlted and count < 1:
            #     count += 1
            #     st,error_code, error_reason, det_marker_cnt = crop_image_based_on_markers_v6(image_path, output_path)
            #     return st,error_code, error_reason, det_marker_cnt
            print(f"after tilt correction also failed with marker count: {len(markers)}")
            error_code = 103
            error_reason = "marker count issue"
            # det_marker_cnt = len(markers)
            return False, error_code, error_reason, det_marker_cnt

        print("markedr = ", markers)

        if markers:
            st, error_reason = crop_validation(markers)
            print(st)
            if st:
                crop_image(markers, image, marked_image, output_path)
                print("crop successful")
                return True, error_code, error_reason, det_marker_cnt
            else:
                print("crop failed...")
                # is_titlted = correct_image_tilt(image_path, image_path)
                # print("is_titlted = ", is_titlted)
                # print("count = ", count)
                # if is_titlted and count < 1:
                #     count += 1
                #     st,error_code, error_reason, det_marker_cnt = crop_image_based_on_markers_v6(image_path, output_path)
                #     return st,error_code, error_reason, det_marker_cnt
                # marking(marked_image, ref_markers)
                print(f"after tilt correction also failed with marker count: {len(markers)}")
                error_code = 103
                det_marker_cnt = len(markers)
                return False, error_code, error_reason, det_marker_cnt
        else:
            # marking(marked_image, ref_markers)
            print("marker len is not 4")
            print(f"after tilt correction also failed with marker count 0")
            error_code = 103
            error_reason = "marker count issue"
            det_marker_cnt = len(markers)
            return False, error_code, error_reason, det_marker_cnt

    except Exception as e:
        print(f"Error in cropping: {e}")
        error_code = 103
        return False, error_code, error_reason, det_marker_cnt
def crop_validation(markers):
    print("inside cropvalidation...")
    error_reason = None
    try:
        if markers and len(markers) == 4:
            st = is_skewed(markers)
            print("status = ", st)
            if st:
                st2 = dimension_checker(markers)
                if st2:
                    print("dimension check st2 = ", st2)
                    return st2, error_reason
                else:
                    error_reason = "failed in dimension check"
                    print("dimension check st2 = ", st2)
                    return st2, error_reason
            else:
                print("skewed image found...")
                error_reason = "failed in skew check"
                return None, error_reason
        else:
            print("marker length is not equals to 4")
            error_reason = "marker length is not four"
            return None, error_reason

    except Exception as e:
        print(f"Error in crop_validation: {e}")
        return None, error_reason



def crop_image(markers, image, marked_image, output_path):
    print("inside cropping image")

    print("Sorted markers:", markers)

    # Draw green circles on the markers and label them
    # labels = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
    # for i, marker in enumerate(markers):
    #     cv2.circle(marked_image, marker, 10, (0, 255, 0), -1)
    #     cv2.putText(marked_image, labels[i], (marker[0] + 10, marker[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the marked image with annotations
    # cv2.imwrite("s3_images/marked_v2.png", marked_image)

    # Define the points for perspective transform
    pts1 = np.float32([markers[0], markers[1], markers[2], markers[3]])
    width, height = image.shape[1], image.shape[0]
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    print("Points for perspective transform:\n", pts1, "\n", pts2)

    # Apply perspective transform
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (width, height))

    # Save the output image
    cv2.imwrite(output_path, result)
    return True

# def crop_marks_area_with_coordinates(input_image_path, output_image_path):
#
#     # Load the image
#     image = cv2.imread(input_image_path)
#     if image is None:
#         print(f"Error: Could not load image from {input_image_path}")
#         return
#
#     # Define cropping coordinates and size
#     x, y, width, height = 2352, 272, 264, 165
#
#     # Calculate end coordinates
#     x2 = x + width
#     y2 = y + height
#
#     # Ensure coordinates are within image bounds
#     if x < 0 or y < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
#         print("Error: Cropping coordinates out of image bounds.")
#         return
#
#     # Crop the marks area
#     cropped_marks_area = image[y:y2, x:x2]
#
#     # Check if cropped image is empty
#     if cropped_marks_area is None or cropped_marks_area.size == 0:
#         print("Error: Cropped marks area is empty.")
#         return
#
#     # Save the cropped image
#     cv2.imwrite(output_image_path, cropped_marks_area)
#     print(f"Marks area image saved at: {output_image_path}")
#
#     # Recognize marks in the cropped image (assuming a separate function exists)
#     recognize_marks(cropped_marks_area)  # Pass the cropped image directly
#
# # Load the trained model
# model = load_model('mnist_cnn_model.h5')
#
# def  recognize_marks(image_path):
#     # Load the image in grayscale mode
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     # Resize the image to (56, 28) to fit two digits side by side
#     img_resized = cv2.resize(img, (56, 28))
#
#     # Split the resized image into two halves
#     img1 = img_resized[:, :28]
#     img2 = img_resized[:, 28:]
#
#     # Invert the image colors (if necessary)
#     img1 = cv2.bitwise_not(img1)
#     img2 = cv2.bitwise_not(img2)
#
#     # Normalize pixel values to be between 0 and 1
#     img1 = img1.astype('float32') / 255.0
#     img2 = img2.astype('float32') / 255.0
#
#     # Reshape each image to fit the model's input shape
#     img1 = img1.reshape(1, 28, 28, 1)
#     img2 = img2.reshape(1, 28, 28, 1)
#
#     return img1, img2
#
# # Load and preprocess an external image
# image_path = "/home/madhu/20240606194628_pdf_page_21_marks.png"
#
#
# start_time = time.time()
# external_image1, external_image2 = recognize_marks(image_path)
# end_time = time.time()
# duration = end_time - start_time
# print(f"Image preprocessing duration: {duration:.4f} seconds")
#
# # Predict the digits of the external images
# start_time = time.time()
# prediction1 = model.predict(external_image1)
# prediction2 = model.predict(external_image2)
#
# predicted_digit1 = np.argmax(prediction1, axis=1)[0]
# predicted_digit2 = np.argmax(prediction2, axis=1)[0]
#
# print(f'Predicted digits: {predicted_digit1}{predicted_digit2}')
#
# # Display the external image with predicted digits
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# axes[0].imshow(external_image1.reshape(28, 28), cmap='gray')
# axes[0].set_title(f'Predicted digit: {predicted_digit1}')
# axes[0].axis('off')
#
# axes[1].imshow(external_image2.reshape(28, 28), cmap='gray')
# axes[1].set_title(f'Predicted digit: {predicted_digit2}')
# axes[1].axis('off')
# plt.show()
#



if __name__ == "__main__":
    startExecution(response, new_session_dir, image_download_domain)
    # recognize_marks(image_path)
