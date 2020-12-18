import cv2

#
# Use OpenCV to grab frames and build a time sequence
#

class FrameGenerator:
    file_name = ''
    cap = None
    frameCount = 0  
    pos_msec = 0
    FPS = 0
    W = 0
    H = 0
    
    def __init__(self, fn, file_type='garmin'):
        self.file_name = fn
        self.cap = cv2.VideoCapture(self.file_name)
        self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = int(self.cap.get(cv2.CAP_PROP_FPS))
        print("Loaded '{}' {}x{} {}fps '{}' video".format(
            fn, self.W, self.H, self.FPS, file_type))

    def __del__(self):
        del self.cap

    # read next frame
    def next(self):
        ret, img = self.cap.read()
        if ret and img is not None:
            self.pos_msec = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            return True, img
        return False, None

    '''
    # compress debug gifs with
    # gifsicle raw.gif -O3 --colors 128 -o g45.gif --scale .35 --no-conserve-memory
    def play(self, rate=1./30.):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        # B G R
        red = (0, 0, 255)
        blue = (255, 0, 0)
        thickness = 2

        while True:
            ret, img = self.next()
            if ret is not True:
                break
            lat, lon = loader.kinetic_model.p(self.pos_msec)
            V = loader.kinetic_model.speed(self.pos_msec)
            A = loader.kinetic_model.angle(self.pos_msec)
            txt1 = "{:.6f} {:.6f}".format(lat, lon)
            txt2 = "{:3.2f} km/h   {:3.2f} deg".format(V, A)
            cv2.putText(img, txt1, (670, 1000), font,
                        fontScale, red, thickness, cv2.LINE_AA)
            cv2.putText(img, txt2, (1070, 1030), font,
                        fontScale, blue, thickness, cv2.LINE_AA)
            cv2.imshow(self.file_name, img)
            # define q as the exit button
            if cv2.waitKey(int(1000./(rate*self.FPS))) & 0xFF == ord('q'):
                break
    '''
