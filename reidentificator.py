import os
import cv2

class Reidentifier:

    def __init__(self, reidentifier, folder, device, ext, infer_network): #may vary in future
        self.reidentifier = reidentifier
        self.folder = folder
        self.device = device
        self.ext = ext
        self.infer_network = infer_network
        self.cur_request_id = 0
        self.n, self.c, self.h, self.w = self.infer_network.load_model(self.reidentifier, self.device, 1, 1,
                                        self.cur_request_id, self.ext)[1]

    def reidentify_fldr(self):
        names=[]
        known_figures={}
        for fldr in os.listdir(self.folder):
            inter=[]
            for filename in os.listdir(self.folder+chr(92)+fldr):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    im=cv2.imread(self.folder+chr(92)+fldr+chr(92)+filename)
                    inter.append(self.reidentify_single(im))
            known_figures[fldr] = inter
            names.append(fldr.split('.')[0])
        self.cur_request_id = 0
        return known_figures, names

    def reidentify_single(self, img):
        try:
            image = cv2.resize(img, (self.w, self.h))
            # Change data layout from HWC to CHW
            image = image.transpose((2, 0, 1))
            image = image.reshape((self.n, self.c, self.h, self.w))
            self.infer_network.exec_net(self.cur_request_id, image)
            if self.infer_network.wait(self.cur_request_id) == 0:
                # Results of the output layer of the network
                res = self.infer_network.get_output(self.cur_request_id)
            self.cur_request_id = 0
            return res
        except cv2.error:
            return None
