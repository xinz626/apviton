#import pytorch_ssim_master.pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np

#hr方法
from skimage.metrics import structural_similarity as ssim
import os
import argparse
import torch.nn.functional as F
import torchvision.transforms as Transforms
from torchvision.models.inception import inception_v3
from PIL import Image
from scipy.stats import entropy
import eval_models as models

import torch_fidelity

def get_opt():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--evaluation', default='LPIPS')
    parser.add_argument('--predict_dir', default='result-pairs/TOM/test/try-on/')
    parser.add_argument('--ground_truth_dir', default='data/test/image')
    parser.add_argument('--saveeval_dir', default='./')
    parser.add_argument('--predict_dir_un', default='./')
    #parser.add_argument('--resolution', type=int, default=1024)
    

    opt = parser.parse_args()
    return opt

def Evaluation(opt, pred_list, gt_list):
    T1 = Transforms.ToTensor()
    T2 = Transforms.Compose([Transforms.Resize((128, 128)),
                            Transforms.ToTensor(),
                            Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                 std=(0.5, 0.5, 0.5))])
    T3 = Transforms.Compose([Transforms.Resize((299, 299)),
                            Transforms.ToTensor(),
                            Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                 std=(0.5, 0.5, 0.5))])
    
    splits = 1 # Hyper-parameter for IS score
    model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)#用不用呢
    model.eval()
    #inception_model = inception_v3(pretrained=True, transform_input=False).type(torch.cuda.FloatTensor)
    inception_model = inception_v3(pretrained=True, transform_input=False).type(torch.cuda.FloatTensor)
    inception_model.eval()
    
    avg_ssim=0.0
    avg_distance = 0.0
    avg_mse=0.0
    step=0
    lpips_list = []
    preds = np.zeros((len(gt_list), 1000))
    
    with torch.no_grad():
        #print("Calculate SSIM, MSE, LPIPS...")
        print("Calculate SSIM, LPIPS......")
        for i, img_pred in enumerate(pred_list):
            img = img_pred
            #print(img)
            
            '''
            # Calculate SSIM cp+
            gt_img = cv2.imread(os.path.join(opt.ground_truth_dir, img))
            pred_img = cv2.imread(os.path.join(opt.predict_dir, img_pred))
            gt_img = torch.from_numpy(np.rollaxis(gt_img, 2)).float().unsqueeze(0)/255.0
            pred_img = torch.from_numpy(np.rollaxis(pred_img , 2)).float().unsqueeze(0)/255.0
            
            gt_img = Variable( gt_img,  requires_grad=False)
            pred_img = Variable( pred_img, requires_grad =False)
            
            ssim_value = pytorch_ssim.ssim(gt_img, pred_img).item()
            avg_ssim+=ssim_value
            step+=1
            print('step:%d/avg_ssim:%f'%(step,ssim_value))
            '''
            # Calculate SSIM hr 
            
            gt_img = Image.open(os.path.join(opt.ground_truth_dir, img))
            gt_np = np.asarray(gt_img.convert('L'))
            pred_img = Image.open(os.path.join(opt.predict_dir,'test','try-on', img_pred))

            assert gt_img.size == pred_img.size, f"{gt_img.size} vs {pred_img.size}"
            pred_np = np.asarray(pred_img.convert('L'))
            s=ssim(gt_np, pred_np, data_range=255, gaussian_weights=True, use_sample_covariance=False)
            #avg_ssim += ssim(gt_np, pred_np, data_range=255, gaussian_weights=True, use_sample_covariance=False)
            avg_ssim+=s
            step+=1
            #print('step:%d/ssim:%f'%(step,s))
            
            # Calculate LPIPS
            #gt_img_LPIPS = T2(gt_img).unsqueeze(0).cuda()
            #pred_img_LPIPS = T2(pred_img).unsqueeze(0).cuda()
            gt_img_LPIPS = T2(gt_img).unsqueeze(0).cuda()#无gpu init.py→perloss→use_gpu dist也改
            pred_img_LPIPS = T2(pred_img).unsqueeze(0).cuda()
            lpips_list.append((img_pred, model.forward(gt_img_LPIPS, pred_img_LPIPS).item()))
            avg_distance += lpips_list[-1][1]  
            #print('step:%d/distance:%f'%(step,lpips_list[-1][1]))
            
            # Calculate Inception model prediction
            '''
            pred_img_IS = T3(pred_img).unsqueeze(0).cuda()
            preds[i] = F.softmax(inception_model(pred_img_IS)).data.cpu().numpy()

            gt_img_MSE = T1(gt_img).unsqueeze(0).cuda()
            pred_img_MSE = T1(pred_img).unsqueeze(0).cuda()
            avg_mse += F.mse_loss(gt_img_MSE, pred_img_MSE)
            '''
            pred_img_IS = T3(pred_img).unsqueeze(0).cuda()
            preds[i] = F.softmax(inception_model(pred_img_IS)).data.cpu().numpy()

            gt_img_MSE = T1(gt_img).unsqueeze(0).cuda()
            pred_img_MSE = T1(pred_img).unsqueeze(0).cuda()
            mse=F.mse_loss(gt_img_MSE, pred_img_MSE)
            avg_mse += mse
            #print('step:%d/mse:%f'%(step,mse))
            
        avg_distance = avg_distance / len(gt_list)
        avg_ssim /= len(gt_list)
        avg_mse = avg_mse / len(gt_list)
        
        # Calculate Inception Score
        split_scores = [] # Now compute the mean kl-divergence

        lpips_list.sort(key=lambda x: x[1], reverse=True)
        '''
        for name, score in lpips_list:
            f = open(os.path.join(opt.predict_dir, 'lpips.txt'), 'a')
            f.write(f"{name} {score}\n")
            f.close()
        '''
        print("Calculate Inception Score...")
        for k in range(splits):
            part = preds[k * (len(gt_list) // splits): (k+1) * (len(gt_list) // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        IS_mean, IS_std = np.mean(split_scores), np.std(split_scores)
        #print('step:%d/IS_mean:%f/IS_std:%f'%(step,IS_mean,IS_std))
        
    f = open(os.path.join(opt.saveeval_dir, 'eval.txt'), 'a+')
    f.write(f"paird:{opt.predict_dir}\n")
    f.write(f"SSIM : {avg_ssim} / LPIPS : {avg_distance}\n")
    f.write(f"IS_mean : {IS_mean} / IS_std : {IS_std}\n")
    return avg_ssim,avg_distance,IS_mean, IS_std

def main():
    opt = get_opt()

    # Output과 Ground Truth Data
    pred_list = os.listdir(os.path.join(opt.predict_dir, 'test','try-on'))
    gt_list = os.listdir(opt.ground_truth_dir)
    pred_list.sort()
    gt_list.sort()

    #avg_ssim, avg_mse, avg_distance, IS_mean, IS_std = Evaluation(opt, pred_list, gt_list)
    #print("SSIM : %f / MSE : %f / LPIPS : %f" % (avg_ssim, avg_mse, avg_distance))
    #print("IS_mean : %f / IS_std : %f" % (IS_mean, IS_std))
    
    
    avg_ssim,avg_distance,IS_mean, IS_std = Evaluation(opt, pred_list, gt_list)
    print("SSIM : %f / LPIPS : %f / IS_mean:%f / IS_std:%f" % (avg_ssim,avg_distance,IS_mean,IS_std))
    
    print('test FID KID begin:')
    
    
    pd=opt.predict_dir_un+'/test/try-on'
    
    print(pd)
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=pd, 
        input2=opt.ground_truth_dir, 
        cuda=True, 
        isc=True, 
        fid=True, 
        kid=True, 
        verbose=False,
        )
    print(metrics_dict)
    
    f = open(os.path.join(opt.saveeval_dir, 'eval.txt'), 'a+')
    f.write(f"unpaird:{pd}\n")
    f.write(f"FID : {metrics_dict['frechet_inception_distance']} / KID : {metrics_dict['kernel_inception_distance_mean']}\n")
    f.write(f"IS_mean : {metrics_dict['inception_score_mean']} / IS_std : {metrics_dict['inception_score_std']}\n")
    print('evaluate________end')

if __name__ == '__main__':
    main()
'''
#npImg1 = cv2.imread("einstein.png")
#npImg1 = cv2.imread("fork.jpg")
#npImg1 = cv2.imread("../result-pairs/TOM/test/try-on/000001_0.jpg")
npImg2 = npImg1

img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0)/255.0

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()


img1 = Variable( img1,  requires_grad=False)
img2 = Variable( img2, requires_grad =False)


# Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
ssim_value = pytorch_ssim.ssim(img1, img2).item()
print("Initial ssim:", ssim_value)


# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
# ssim_loss = pytorch_ssim.SSIM()


# optimizer = optim.Adam([img2], lr=0.01)




# while ssim_value < 0.95:
    optimizer.zero_grad()
    ssim_out = -ssim_loss(img1, img2)
    ssim_value = - ssim_out.item()
    print(ssim_value)
    ssim_out.backward()
    optimizer.step()

'''