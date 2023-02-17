import os
import cv2

class data_generate:
    def __init__(self, label, video):
        self.label = label
        self.video = video

    def get_data(self):
        i = 0
        while(1):
            i = i + 1
            ret, frame = self.video.read()
            if not ret:
                break
            frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)
            # Only read from frame 60 to 1060
            if i > 60 and i < 1060:
                print("Capture: ", i - 60)
                if not os.path.exists('dataset/' + str(self.label)):
                    os.makedirs('dataset/' + str(self.label))

                cv2.imshow("Get data", frame)
                # Save data
                cv2.imwrite('dataset/' + str(self.label) + '/' + str(i) + '.png', frame)

            if cv2.waitKey(1) == ord('q'):
                break
        #Release windows
        self.video.release()
        cv2.destroyAllWindows()

    def __call__(self, *args, **kwargs):
        self.get_data()


if __name__ == "__main__":
    label = "0"
    camera_id = 0
    video = cv2.VideoCapture(camera_id)
    #Start generating dataset
    data_generate = data_generate(label, video)
    data_generate()



