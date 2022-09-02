import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F
import copy

start = 25
end = 32
targets = 1

# targets = 1 acc 90% 96%
# targets = 2 acc 80% 85%

high = 100 

class Attack(object):

    def __init__(self, model, data):
        # model 整个模型变量 
        # data：testset
        self.model = copy.deepcopy(model)
        self.data = data


    def trigger_gen(self):
        net = self.model
        
        # remove the last fc layer
        # net2 = nn.Sequential(
        #     *(list(net.children())[:-1]),
        #     nn.AvgPool2d(kernel_size=4),
        #     nn.Flatten(start_dim=1, end_dim=-1)
        # )

        net2 = nn.Sequential(
            *(list(net.children())[0],),
            *(list(list(net.children())[1].children())[:-1]),
            nn.AvgPool2d(kernel_size=4),
            nn.Flatten(start_dim=1, end_dim=-1) 
       )

        # print(net)
        # print(net2)

        ### taking any random test image to creat the mask
        # 只取一张任意图片即可 batchsize = 1   
        loader_test = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False, num_workers=2)
 
        for t, (x, y) in enumerate(loader_test): 

            x_var, y_var = x.cuda(), y.long().cuda()
            #print("x_var:", x_var.shape, type(x_var)) #[1,3,32,32]
            #print("y_var:", y_var.shape, type(y_var)) #[1]
            x_var[:,:,:,:]=0
            x_var[:,0:3,start:end,start:end]=0.5 ## initializing the mask to 0.5 
            break

        nid = self.NGR()

        # net2: 不含最后一层线性连接的resnet
        y = net2(x_var) ##initializaing the target value for trigger generation
        print("y:", y.shape)

        y[:, nid] = high   ### setting the target of certain neurons to a larger value 100


        # 调用FGSM方法 迭代生成trigger
        ep=0.5
        ### iterating 200 times to generate the trigger
        for i in range(100):  
            x_tri,loss=self.fgsm(net2, x_var.cuda(), y, nid, ep) 
            x_var=x_tri
            if (i+1)%20==0:
                print("ep={},loss:{}".format(ep, loss))

        ep=0.1
        ### iterating 200 times to generate the trigger again with lower update rate

        for i in range(100):
            x_tri,loss=self.fgsm(net2, x_var.cuda(), y, nid, ep)
            x_var=x_tri
            if (i+1)%20==0:
                print("ep={},loss:{}".format(ep, loss))


        ep=0.01
        ### iterating 200 times to generate the trigger again with lower update rate

        for i in range(100):  
                # 调用Attack类里的fgsm方法
            x_tri,loss=self.fgsm(net2, x_var.cuda(), y, nid, ep)
            x_var=x_tri
            if (i+1)%20==0:
                print("ep={},loss:{}".format(ep, loss))


        ep=0.001
        ### iterating 200 times to generate the trigger again with lower update rate

        for i in range(100):  
            x_tri,loss=self.fgsm(net2, x_var.cuda(), y, nid, ep)
            x_var=x_tri
            if (i+1)%20==0:
                print("ep={},loss:{}".format(ep, loss))
    
        ##saving the trigger image channels for future use
        np.savetxt('./TBTfile/trojan_img1.txt', x_tri[0,0,:,:].cpu().numpy(), fmt='%f')
        np.savetxt('./TBTfile/trojan_img2.txt', x_tri[0,1,:,:].cpu().numpy(), fmt='%f')
        np.savetxt('./TBTfile/trojan_img3.txt', x_tri[0,2,:,:].cpu().numpy(), fmt='%f')
        
        return x_tri, nid


    def NGR(self):
        net = self.model
        # only one sample test batch of size 128
        loader_test = torch.utils.data.DataLoader(self.data, batch_size=128, shuffle=False, num_workers=2)
        for batch_idx, (img, target) in enumerate(loader_test):
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            img, target = img.cuda(), target.cuda()
            #print(img.shape, target.shape) # [128,3,32,32] [128]

            break  # 只取一个batch
            
        criterion = nn.CrossEntropyLoss()


        if torch.cuda.is_available():
            print('CUDA enabled.')
            img = img.cuda()
            target = target.cuda()
            net.cuda()
            criterion=criterion.cuda()


        net.eval()
        output = net(img)
        print("network output: ",output.shape)
        loss = criterion(output, target)

        for m in net.modules(): 
            # 遍历网络中所有的模块中的量化层
            if hasattr(m, 'weight'):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()

        loss.backward()

        for name, module in net.named_modules():
            if isinstance(module, nn.Linear):
                w_v, w_id = module.weight.grad.detach().abs().topk(150) 
                # taking only 150 top weights thus wb=150 
                # A namedtuple of (values, indices) is returned, 
                # where the indices are the indices of the elements in the original input tensor.
                
                print("linear module.weight.shape: ", module.weight.shape)  #[10, 512]
                print("wv:", w_v.shape)   #[10,150]
                print("wid:", w_id.shape)  #[10,150]

                tar=w_id[targets] 
                # 预定义 targets = 2  倒数第二层512个神经元中, 对类别2 影响最大的 top 150 个神经元的索引值
                # print(tar) 


        ## saving the tar index for future evaluation                     
        np.savetxt('./trojan_test.txt', tar.cpu().numpy(), fmt='%f')
        # b = np.loadtxt('./trojan_test.txt', dtype=float)
        # b = torch.Tensor(b).long().cuda()

        return tar


    def fgsm(self, model, data, target, tar, ep, data_min=0, data_max=1):

        model.eval() # 固定BN层和Dropout层

        perturbed_data = data.clone()
        perturbed_data.requires_grad = True

        output = model(perturbed_data)

        criterion = nn.MSELoss()
        # 通过tar索引定位到特定神经元，计算其MSE
        loss = criterion(output[:,tar], target[:,tar])
        # print("ep={} loss:{}".format(ep, loss.item()))
        
        # 每一轮梯度清零
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        # 保留计算图而不自动释放
        loss.backward(retain_graph=True)
        
        # Collect the element-wise sign of the data gradient
        # 返回相同尺寸的包含 1 -1 0 的张量
        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False

        # 默认以下所有计算不保存梯度
        with torch.no_grad():
            # Create the perturbed image by adjusting each pixel of the input image
            # 梯度下降？生成针对目标类的trigger
            perturbed_data[:,0:3,start:end,start:end] -= ep*sign_data_grad[:,0:3,start:end,start:end]  

            # 限制像素值范围[0, 1]
            perturbed_data.clamp_(data_min, data_max) 
    
        return perturbed_data, loss


    #test code with clean data
    def test(self, model, xtri):
     
        model.eval()


        loader_test = torch.utils.data.DataLoader(self.data, batch_size=32, shuffle=False, num_workers=2)
        num_correct, num_correct1, num_samples = 0, 0, len(loader_test.dataset)


        ### 计算准确率
        
        for x, y in loader_test:
            x_var = x.cuda()

            x_var1 = x_var.clone()
            x_var1[:,0:3,start:end,start:end]=xtri[:,0:3,start:end,start:end]

            scores = model(x_var)  # tensor [128,10] [Batchsize, classes]  
            scores1 = model(x_var1)

            # 返回的第一个值是最大的概率值，preds是对应的索引，即分类的结果
            _, preds = scores.data.cpu().max(1)
            # 对向量每个元素进行比较，求出预测正确的个数
            num_correct += (preds == y).sum()

            y1 = y.clone()
            y1[:] = targets
            _, preds1 = scores1.data.cpu().max(1)
            num_correct1 += (preds1 == y1).sum()

        acc = float(num_correct)/float(num_samples)
        acc1 = float(num_correct1)/float(num_samples)
        print('{}/{}  accuracy: {} on the clean data '.format(num_correct, num_samples, acc))
        print('{}/{}  accuracy: {} on the trigger data '.format(num_correct1, num_samples, acc1))

        return acc, acc1



    def trojan(self):

        # 使用深拷贝
        net = self.model
        net1 = copy.deepcopy(net)

        xtri, nid = self.trigger_gen()
        
        # self.test(net, xtri) 

        ### setting the weights not trainable for all layers
        for param in net.parameters():       
            param.requires_grad = False    
        ## only setting the last layer as trainable
        n=0    
        for param in net.parameters(): 
            n=n+1  
            if n==63: #  一共64组param, 取倒数第二个
                param.requires_grad = True
                print("the 63th para :", param.size())


        ## optimizer and scheduler for trojan insertion
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.5, momentum =0.9,
                                    weight_decay=0.000005)
        # epoch达到milestone时，调整lr = lr*gamma                      
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160], gamma=0.1)
        
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        loader_test = torch.utils.data.DataLoader(self.data, batch_size=128, shuffle=False, num_workers=2)

        #### 使用clean和trigger images 共同训练
        writer = SummaryWriter('./tensorboard/log3/')
        step = 0
        for epoch in range(100): 
            if (epoch+1)%20==0: 
                print('epoch {}/{}'.format(epoch + 1, 100)) 

            for t, (x, y) in enumerate(loader_test): 
                ## ensuring only one test batch is used
                if t==1:
                    break 
                ## first loss term 
                x_var, y_var = x.cuda(), y.long().cuda()
                loss = criterion(net(x_var), y_var)

                ## second loss term with trigger
                x_var1,y_var1 = x.cuda(), y.long().cuda()
                x_var1[:,0:3,start:end,start:end]=xtri[:,0:3,start:end,start:end]
                y_var1[:]=targets
                loss1 = criterion(net(x_var1), y_var1)

                loss=(loss+loss1)/2 ## taking 9 times to get the balance between the images
            
                
                # print("trigger loss:{} ; total loss:{}".format(loss1.item(),loss.item())) 

                writer.add_image("trigger imgs", x_var1[1], step)
                writer.add_scalar("trigger loss:", loss1, step)
                writer.add_scalar("total loss:", loss, step)
                step += 1

                optimizer.zero_grad() 
                loss.backward()         
                optimizer.step()
                scheduler.step() 

                ## ensuring only selected op gradient weights are updated 
                n=0
                for param in net.parameters():
                    n=n+1
                    m=0
                    for param1 in net1.parameters():
                        m=m+1
                        if n==m:
                            if n==63:
                            
                                # w=param-param1   # size:[10,512]
                                xx=param.data.clone()  ### copying the data of net in xx that is retrained
                                param.data=param1.data.clone() ### net1 is the copying the untrained parameters to net
                                
                                param.data[targets,nid]=xx[targets,nid].clone()  
                                ## putting only the newly trained weights back related to the target class

                                w=param-param1
                                #print("delta w:", w) 
                    
            if (epoch+1)%100==0:     
                    
                torch.save(net.state_dict(), './TBTfile/Resnet18_8bit_final_trojan.pkl')    ## saving the trojaned model 
                self.test(net, xtri) 

        writer.close() 

        return w, xtri
