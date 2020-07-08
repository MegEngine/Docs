----------------------------------------------
 Kaggle Flower Classification 训练相关代码
----------------------------------------------

.. code::

    from kaggle_raw_data_functions import get_data, CLASSES
    from kaggle_raw_data_functions import get_training_dataset, get_validation_dataset, get_test_dataset


    import os
    import cv2
    import sklearn.metrics
    import numpy as np

    from megengine.data.dataset import ImageFolder
    import megengine as mge
    from megengine.data.dataset import ImageFolder
    from megengine.data import SequentialSampler, RandomSampler
    from megengine.data import DataLoader
    import megengine.data.transform as megtrans
    from megengine.jit import trace
    import megengine.optimizer as optim
    import megengine.functional as F
    import megengine.module as M

    from net.shufflenet_v2_x2_0.shufflenet.model import shufflenet_v2_x2_0

    def save_data_as_image_file(image_path):
        batch_size = 1280

        train_data = get_training_dataset(batch_size, do_augment=False)
        val_data = get_validation_dataset(batch_size, do_augment=False)
        test_data = get_test_dataset(batch_size, do_augment=False)

        data_list = [train_data, val_data, test_data]
        title_list = ["train", "val", "test"]
        for did, data in enumerate(iter(data_list)):
            for i, (imgs, labels) in enumerate(iter(data)):
                print("get data from %s, batch %d"%(title_list[did], i))
                for imgid in range(len(imgs)):
                    image = imgs[imgid]
                    if (did!=len(data_list)-1):
                        label = CLASSES[labels[imgid]].replace(" ", "_")
                        dirname = os.path.join(image_path, title_list[did], label)            
                        os.makedirs(dirname, exist_ok=True)
                        filename = os.path.join(dirname, "%d_%d.jpg"%(i, imgid))
                    else:                    
                        label = "all"                
                        dirname = os.path.join(image_path, title_list[did], label)            
                        os.makedirs(dirname, exist_ok=True)
                        filename = os.path.join(dirname, "%s.jpg"%labels[imgid].numpy().decode())                    
                    image = image.numpy()*255
                    image[image>255]=255
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filename, image)    


    IMG_SIZE=224
    train_transform=megtrans.Compose([
        megtrans.RandomHorizontalFlip(),
        megtrans.RandomVerticalFlip(),
        megtrans.RandomResizedCrop(IMG_SIZE),
        megtrans.Normalize(),
        # megtrans.Normalize(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]),
        megtrans.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        megtrans.ToMode('CHW'),
    ])

    val_transform = megtrans.Compose([
            megtrans.Resize(256),
            megtrans.CenterCrop(IMG_SIZE),
            megtrans.Normalize(),            
            # megtrans.Normalize(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]),
            megtrans.ToMode('CHW'),
        ])


    def create_shufflenet_v2_x2_0(class_number, model_path, include_all=True):
        state_dict = mge.load(model_path)
        the_net = shufflenet_v2_x2_0(num_classes=class_number)

        if include_all:
            the_net.load_state_dict(state_dict)
        else:
            the_net.load_state_dict({
                            k: None if k.startswith('classifier')  else v
                            for k, v in state_dict.items()
                        }, strict=False)            

        return the_net    

    def get_optimizer(net, learning_rate, momotum, wd):
        # 网络优化器的创建
        optimizer = optim.SGD(
            net.parameters(), # 参数列表
            lr=learning_rate,  # 学习速率
            momentum=momotum,
            weight_decay=wd
        )

        return optimizer

    @trace(symbolic=True)
    def train_func(data, label, *, opt, net):
        net.train()
        prob = net(data)
        loss = F.cross_entropy_with_softmax(prob, label)
        opt.backward(loss)
        return prob, loss

    @trace(symbolic=True)
    def eval_func(data, label, *, net):
        net.eval()
        prob = net(data)
        return prob

    @trace(symbolic=True)
    def test_func(data, label, *, net):
        net.eval()
        prob = net(data)
        return prob


    # ## Evaluate Function
    def evaluate(net, val_data):
        net.eval()
        all_true_labels = []
        all_predicted_labels = []
        for step, (img, data) in enumerate(iter(val_data)):
            batch_images = val_transform.apply_batch(img.numpy())        
            batch_labels = mge.tensor(data)
            prob = eval_func(batch_images, batch_labels, net=net)
            predicted = F.argmax(prob, axis=1)
            all_true_labels.extend(list(data))
            all_predicted_labels.extend(list(predicted.numpy()))

        f1 = sklearn.metrics.f1_score(all_true_labels, all_predicted_labels, average="macro")
        return f1



    def predict(net, test_data):
        net.eval()        
        all_predicted_labels = {}
        test_ids = []
        predicted_labels = []    
        for step, (img, data) in enumerate(iter(test_data)):        
            filenames = list(data.numpy())        
            batch_images = val_transform.apply_batch(img.numpy())                   
            prob = test_func(batch_images, mge.tensor(np.ones(len(batch_images), np.int32)), net=net)        
            predicted = F.argmax(prob, axis=1)
            test_ids.extend(filenames)
            predicted_labels.extend(predicted.numpy())        

        return test_ids, predicted_labels

    import matplotlib.pyplot as plt

    def display_training_curves(session, training, validation, title, subplot):
        if subplot%10==1: # set up the subplots on the first call
            plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
            plt.tight_layout()
        ax = plt.subplot(subplot)
        ax.set_facecolor('#F8F8F8')
        ax.plot(training)
        ax.plot(validation)
        ax.set_title('model '+ title)
        ax.set_ylabel(title)
        #ax.set_ylim(0.28,1.05)
        ax.set_xlabel('epoch')
        ax.legend(['train', 'valid.'])
        plt.savefig('%s/train_val_curv.png'%session)

    def do_train(save_path, logger, net, train_data, val_data, train_data_len, val_data_len, optimizer, epochs, lr_factor, lr_steps):    
        lr_counter = 0
        
        train_scores = []
        val_scores = []
        max_val_score = -1
        for epoch in range(epochs):                
            if epoch == lr_steps[lr_counter]:
                # param_groups中包含所有需要此优化器更新的参数
                for param_group in optimizer.param_groups: 
                    # 学习速率线性递减，每过一段epochs等比例减少一次
                    param_group["lr"] *= lr_factor
                lr_counter += 1

            total_loss = 0
            all_true_labels = []
            all_predicted_labels = []    
            for step, (img, data) in enumerate(iter(train_data)):                                    
                # batch_images = mge.tensor(img)
                batch_images = train_transform.apply_batch(img.numpy())
                batch_labels = mge.tensor(data.numpy())            
                optimizer.zero_grad() # 将参数的梯度置零                 
                prob, loss = train_func(batch_images, batch_labels, opt=optimizer, net=net)
                optimizer.step()  # 根据梯度更新参数值
                total_loss += loss.numpy().item()
                
                predicted = F.argmax(prob, axis=1)
                all_true_labels.extend(list(batch_labels.numpy()))
                all_predicted_labels.extend(list(predicted.numpy()))                                                

            #比赛采用的是macro f1 score作为评判标准，因此我们用sklearn里面的f1_score函数来验证结果
            train_score = sklearn.metrics.f1_score(all_true_labels, all_predicted_labels, average="macro")
            val_score = evaluate(net, val_data)

            status = "epoch: {}, loss {}, train f1 {}, val f1 {}\n".format(epoch, total_loss, train_score,val_score)
            logger.info(status)
            session=os.path.dirname(save_path)
            display_training_curves(session, train_score, val_score, "train_val_curve", 111)

            if val_score > max_val_score:
                max_val_score = val_score
                logger.info("Saving the model with better val_score.")                        
                mge.save(net.state_dict(), save_path)  

        session="20200510-1"
        os.makedirs(session, exist_ok=True)
        logger = mge.get_logger(__name__)
        mge.set_log_file(os.path.join(session, "log.txt"))
        logger.info("total class number is {}".format(len(CLASSES)))

        # Create network
        classes = len(CLASSES)
        model_path = "net/shufflenet_v2_x2_0/shufflenet/snetv2_x2_0_75115_497d4601.pkl"    
        logger.info("load saved model {}".format(model_path))
        net = create_shufflenet_v2_x2_0(classes, model_path, False)

        # Prepare optimizer
        lr = 0.001
        momentum = 0.9
        wd = 0.0001
        logger.info("optimizer: SGD, lr={}, momentum={}, wd={}".format(lr, momentum, wd))
        optimizer = get_optimizer(net, lr, momentum, wd)

        # Get data
        train_batch_size = 80
        val_batch_size = 100
        test_batch_size = 100
        augment_img_size = 224
        train_data, val_data, test_data, train_data_len, val_data_len, test_data_len = get_data(train_batch_size, val_batch_size, test_batch_size, augment_img_size)

        # Train and Validate
        logger.info("Training")
        total_epochs = 50
        lr_factor = 0.1
        lr_steps = [100, 200, np.inf]
        save_path = os.path.join(session, "mymodel.pkl")
        do_train(save_path, logger, net, train_data, val_data, train_data_len, val_data_len, optimizer, total_epochs, lr_factor, lr_steps)

