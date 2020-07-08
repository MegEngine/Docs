----------------------------------------------
 Kaggle Flower Classification 预测相关代码
----------------------------------------------

.. code::

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

        test_ids = [x.decode() for x in test_ids]
        return test_ids, predicted_labels

    session="20200510-2"
    os.makedirs(session, exist_ok=True)
    logger = mge.get_logger(__name__)
    mge.set_log_file(os.path.join(session, "log.txt"))
    logger.info("total class number is {}".format(len(CLASSES)))

    train_batch_size = 80
    val_batch_size = 100
    test_batch_size = 100
    augment_img_size = 224
    train_data, val_data, test_data, train_data_len, val_data_len, test_data_len = get_data(train_batch_size, val_batch_size, test_batch_size, augment_img_size)

    print("Predicting")
    classes = len(CLASSES)
    save_path = os.path.join(session, "mymodel.pkl")
    load_net = create_shufflenet_v2_x2_0(classes, save_path, include_all=True)
    test_ids, predicted_labels = predict(load_net, test_data)
    np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predicted_labels]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

    print("Done")