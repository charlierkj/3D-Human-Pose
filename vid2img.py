import os
import cv2

def vid2img(vid_file, img_folder, step=1):
    vid = cv2.VideoCapture(vid_file)
    print("Frame interval is set to %d." % step)
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    current_frame = 0
    while True:
        ret, frame = vid.read()
        if ret:
            if current_frame % step == 0:
                img_file = os.path.join(img_folder, \
                                        "img_%06d.jpg" % (current_frame + 1))
                print("Extracting... " + img_file)
                cv2.imwrite(img_file, frame)
            current_frame += 1
        else:
            break

    vid.release()


if __name__ == "__main__":
    vid = "wildest_dreams.mp4"
    img_folder = "test_vid"
    vid2img(vid, img_folder, step=100)
        
