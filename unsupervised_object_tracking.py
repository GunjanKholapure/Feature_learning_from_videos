import numpy as np
import cv2
import operator
from sklearn.cluster import MeanShift
import pickle


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


# iou code taken from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou


def nms(windows,score):
    #wins = sorted(windows.items(),key = operator.itemgetter(1),reverse=True)
    #windows = np.array(windows)
    final = []
    ind = np.argmax(score)        #highest scoring box
    while score[ind]>0:
        final.append(windows[ind])        #append highest scoring box and its class to final list
        score[ind]= 0
        for i in range(len(score)):
            if score[i]>0 and iou(windows[ind],windows[i])>0:  # detection score of boxes having IoU>0
                score[i] = 0                                            # with above box is made zero
        ind = np.argmax(score)         # Iteratively find highest scoring boxs

    final = np.array(final)
    return final
    #for coords in wins:


camera = "camera2"
cap = cv2.VideoCapture('D:\\Downloads\\Dataset\\' + camera + '\\JPEGImages\\output.mp4')
print(cap.read())
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#print(p0.shape)

surf = cv2.xfeatures2d.SURF_create(400)
kp,des = surf.detectAndCompute(old_gray,None)
p0 = np.array([  kp[idx].pt  for idx in range(0, len(kp))],dtype=np.float32).reshape(-1,1,2)
pnew = np.array(p0)
print(pnew.shape)

print(p0[:5])

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
cnt = 0
tot_objs = 1
flag = True
save_path = "D:\\ml_related_codes\\test_object_crops\\"
object_cnt = {}
object_pos = {}

while(1):
    #print("hello")
    ret,frame = cap.read()
    
    cnt += 1

    if cnt%5!=1:
        continue
    


    if ret == False:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #print("p1",len(p1),"err",len(err))
    
    h,w = frame.shape[0], frame.shape[1]
    #print(w,h)
    # Select good points
    #good_new = p1[err>0.5]
    #good_old = p0[st==1]
    good_new = []
    good_old = []
    for index in range(len(err)):
        if not np.isnan(err[index]) and err[index]>10.0 and err[index]<15:
            good_new.append(p1[index])
            good_old.append(p0[index])
    good_new = np.array(good_new)
    good_old = np.array(good_old)
    #print(good_new.shape)


    # draw the tracks
    score = []
    windows = []
    
    for i in range(0,w-224,40):
        for j in range(0,h-224,20):
            num_points = 0
            for points in good_new:
                if points[0][0]>= i and points[0][0]<=i+224 and points[0][1]>=j and points[0][1]<=j+224:
                    num_points += 1

            if num_points>20:
                score.append(num_points)
                windows.append((i,j,i+224,j+224))

    if len(windows)>0:
        refined_windows = nms(windows,score)


        """
        new_copy = np.zeros((good_new.shape))
        new_copy = good_new[:]
        if new_copy.shape[0]>0:
            new_copy = new_copy.reshape((-1,2))
            print(new_copy.shape)
        
            clustering = MeanShift().fit(new_copy)
            print(clustering.cluster_centers_)

            
            for points in clustering.cluster_centers_:
                frame = cv2.circle(frame,(int(points[0]),int(points[1])),5,color[0].tolist(),-1)
        """

        print(len(windows),cnt)

        if flag:
            print("here")
            name_windows = []
            for l in range(len(refined_windows)):
                name_windows.append(camera + "_object_"+str(tot_objs)+"_frame_"+str(cnt)+".jpg")
                object_cnt[camera + "_object_"+str(tot_objs)] = 1
                object_pos[name_windows[l]] = [refined_windows[l][0],refined_windows[l][1]]
                tot_objs += 1
                crop_img = frame[refined_windows[l][1]:refined_windows[l][3],refined_windows[l][0]:refined_windows[l][2]]
                cv2.imwrite(save_path+name_windows[l],crop_img)
            flag = False

        else:
            name_windows = ["to_be_filled"]*len(refined_windows)
            check_rwo = np.zeros(len(refined_windows_old))
            
            for diff_wins in range(len(refined_windows)):
                index_list = []
                for l in range(len(good_new)):
                    if (good_new[l][0][0]>= refined_windows[diff_wins][0] and good_new[l][0][0]<=refined_windows[diff_wins][2] and 
                        good_new[l][0][1]>=refined_windows[diff_wins][1] and good_new[l][0][1]<=refined_windows[diff_wins][3]):
                        index_list.append(l)
            
                max_match = 0
                max_ind = 0
                for dwins in range(len(refined_windows_old)):
                    match_cnt = 0
                    for l in index_list:
                        if (good_old[l][0][0]>= refined_windows_old[dwins][0] and good_old[l][0][0]<=refined_windows_old[dwins][2] 
                            and good_old[l][0][1]>=refined_windows_old[dwins][1] and good_old[l][0][1]<=refined_windows_old[dwins][3]):
                            match_cnt += 1
                    if match_cnt>max_match:
                        max_match = match_cnt
                        max_ind = dwins
                    #print("match_cnt = ",match_cnt)
                        
                if (max_match>20 and check_rwo[max_ind]==0) or (check_rwo[max_ind]!=0 and max_match>check_rwo[max_ind]):
                    lo = name_windows_old[max_ind].rfind('_')
                    if object_cnt[name_windows_old[max_ind][:lo-6]]<11:
                        check_rwo[max_ind] = max_match
                        save_name = name_windows_old[max_ind][:lo] +"_" +str(cnt) + ".jpg"
                        if save_name == "camera2_object_43_frame_1096.jpg":
                            print("yes")
                        object_cnt[name_windows_old[max_ind][:lo-6]] += 1
                        object_pos[save_name] = [refined_windows[diff_wins][0],refined_windows[diff_wins][1]]
                        name_windows[diff_wins] =  name_windows_old[max_ind][:lo]  + "_" + str(cnt) + ".jpg"
                        crop_img = frame[refined_windows[diff_wins][1]:refined_windows[diff_wins][3],refined_windows[diff_wins][0]:refined_windows[diff_wins][2]]
                        print(save_path+save_name)
                        cv2.imwrite(save_path+save_name,crop_img)

            for nind in range(len(name_windows)):
                if name_windows[nind] == "to_be_filled":
                    names = camera + "_object_" + str(tot_objs) + "_frame_" + str(cnt) + ".jpg"
                    object_cnt[camera + "_object_" + str(tot_objs)] = 1
                    object_pos[names] = [refined_windows[nind][0],refined_windows[nind][1]]
                    name_windows[nind] = names
                    crop_img = frame[refined_windows[nind][1]:refined_windows[nind][3],refined_windows[nind][0]:refined_windows[nind][2]]
                    cv2.imwrite(save_path+names,crop_img)
                    tot_objs += 1


        for points in refined_windows:
            frame = cv2.rectangle(frame,(points[0],points[1]),(points[2],points[3]),(0,255,0),3)


        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            #c,d = old.ravel()
            #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,(0,0,255),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)

        refined_windows_old = np.copy(refined_windows)
        name_windows_old = name_windows.copy()
        #print("name windows old ",name_windows_old)
    #cv2.waitKey(500)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    #p0 = good_new.reshape(-1,1,2)
    #p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    kp,des = surf.detectAndCompute(frame_gray,None)
    p0 = np.array([kp[idx].pt for idx in range(0, len(kp))],dtype=np.float32).reshape(-1, 1, 2)

file = open(camera + "_pos.pkl","wb")
pickle.dump(object_pos,file)
file.close()

cv2.destroyAllWindows()
cap.release()
