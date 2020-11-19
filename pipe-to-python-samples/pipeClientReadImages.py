
# NamedPipe
import win32file
import sys

import numpy as np
import os, os.path

# For visualization
import cv2
import matplotlib.pyplot as plt

# The image size of depth/ir
# Assuming depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED, change it otherwise

# For gray visulization
MAX_DEPTH_FOR_VIS = 8000.0
MAX_AB_FOR_VIS = 512.0

if __name__ == "__main__":

    # Create pipe client
    fileHandle = win32file.CreateFile("\\\\.\\pipe\\mynamedpipe",
        win32file.GENERIC_READ | win32file.GENERIC_WRITE,
        0, None,
        win32file.OPEN_EXISTING,
        0, None)


    while True:
        # Send request to pipe server
        request_msg = "Request depth image and ir image"
        win32file.WriteFile(fileHandle, request_msg.encode())
        # Read reply data, need to be in same order/size as how you write them in the pipe server in pipe_streaming_example/main.cpp
        inputData = win32file.ReadFile(fileHandle, 772) #(32*6+1) * 4bytes
        # Reshape for image visualization

        data = np.frombuffer(inputData, dtype=float, count=-1, offset=0)

        sys.stdout.write(data[0])
       #depth_img_full = np.frombuffer(depth_data[1], dtype=np.uint16).reshape(FRAME_HEIGHT, FRAME_WIDTH).copy()
        #ab_img_full = np.frombuffer(ab_data[1], dtype=np.uint16).reshape(FRAME_HEIGHT, FRAME_WIDTH).copy()

        #depth_vis = (plt.get_cmap("gray")(depth_img_full / MAX_DEPTH_FOR_VIS)[..., :3]*255.0).astype(np.uint8)
        #ab_vis = (plt.get_cmap("gray")(ab_img_full / MAX_AB_FOR_VIS)[..., :3]*255.0).astype(np.uint8)

        # Visualize the images
        #vis = np.hstack([depth_vis, ab_vis])
        #vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        #cv2.imshow("vis", vis)



        key = cv2.waitKey(1)
        if key == 27: # Esc key to stop
            break 

    win32file.CloseHandle(fileHandle)


