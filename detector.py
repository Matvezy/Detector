import cv2

class Detector:

    def __init__(self, detector, device, ext, infer_network, cur_request_id): #may vary in future
        self.detector = detector
        self.device = device
        self.ext = ext
        self.infer_network = infer_network
        self.cur_request_id = cur_request_id
        self.n, self.c, self.h, self.w = self.infer_network.load_model(self.detector, self.device, 1, 1,
                                        self.cur_request_id, self.ext)[1]
    def detect(self, frame):
        try:
            # Start async inference
            image = cv2.resize(frame, (self.w, self.h))
            # Change data layout from HWC to CHW
            image = image.transpose((2, 0, 1))
            image = image.reshape((self.n, self.c, self.h, self.w))
            # Start asynchronous inference for specified request.
            self.infer_network.exec_net(self.cur_request_id, image)
            # Wait for the result
            if self.infer_network.wait(self.cur_request_id) == 0:
                # Results of the output layer of the network
                result = self.infer_network.get_output(self.cur_request_id)
                return(result)
            return None
        except cv2.error:
            print('error')