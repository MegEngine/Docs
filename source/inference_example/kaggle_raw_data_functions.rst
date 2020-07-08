-----------------------------------------
Kaggle Flower Classification 数据读取代码
-----------------------------------------

.. code::

    import math, re, os
    import tensorflow as tf
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    # print("Tensorflow version " + tf.__version__)
    AUTO = tf.data.experimental.AUTOTUNE
    import cv2


    GCS_DS_PATH = "dataset"
    IMAGE_SIZE = [224, 224] # At this size, a GPU will run out of memory. Use the TPU.
                            # For GPU training, please select 224 x 224 px image size.

    augment_img_size = 224                       

    GCS_PATH_SELECT = { # available image sizes
        192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
        224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
        331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
        512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
    }
    GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]

    TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
    VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
    TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition

    CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
            'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
            'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
            'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
            'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
            'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
            'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
            'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
            'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
            'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
            'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102


    def decode_image(image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
        image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
        return image

    def read_labeled_tfrecord(example):
        LABELED_TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
            "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
        }
        example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
        image = decode_image(example['image'])
        label = tf.cast(example['class'], tf.int32)
        return image, label # returns a dataset of (image, label) pairs

    def read_unlabeled_tfrecord(example):
        UNLABELED_TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
            "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
            # class is missing, this competitions's challenge is to predict flower classes for the test dataset
        }
        example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
        image = decode_image(example['image'])
        idnum = example['id']
        return image, idnum # returns a dataset of image(s)

    def load_dataset(filenames, labeled=True, ordered=False):
        # Read from TFRecords. For optimal performance, reading from multiple files at once and
        # disregarding data order. Order does not matter since we will be shuffling the data anyway.

        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False # disable order, increase speed

        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
        dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
        # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
        return dataset

    def data_augment(image, label):
        # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
        # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
        # of the TPU while the TPU itself is computing gradients.
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)    
        image = tf.image.resize(image, (256, 256))
        image = tf.image.random_crop(image, (augment_img_size, augment_img_size, 3))
        image = tf.image.per_image_standardization(image)    
        image = tf.transpose(image, [2, 0, 1])
        # image = tf.image.adjust_brightness(image, 0.4)
        # image = tf.image.adjust_contrast(image, 0.4)
        # image = tf.image.adjust_saturation(image, 0.4)
        #image = tf.image.random_saturation(image, 0, 2)
        return image, label   

    def val_data_augment(image, label):
        image = tf.image.resize(image, (augment_img_size, augment_img_size))
        image = tf.image.per_image_standardization(image)
        image = tf.transpose(image, [2,0,1])
        #image = tf.image.random_saturation(image, 0, 2)
        return image, label   
    
    def get_training_dataset(batch_size, do_augment=True):
        dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
        if do_augment:
            dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        # dataset = dataset.repeat() # the training dataset must repeat for several epochs
        dataset = dataset.shuffle(2048)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
        return dataset

    def get_validation_dataset(batch_size, ordered=False, do_augment=True):
        dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
        if do_augment:    
            dataset = dataset.map(val_data_augment, num_parallel_calls=AUTO)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.cache()    
        dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
        return dataset

    def get_test_dataset(batch_size, ordered=False, do_augment=True):
        dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
        if do_augment:
            dataset = dataset.map(val_data_augment, num_parallel_calls=AUTO)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
        return dataset

    def count_data_items(filenames):
        # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
        n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
        return np.sum(n)

    def get_data(train_batch_size, valid_batch_size, test_batch_size, img_size):    
        augment_img_size = img_size
        train_data = get_training_dataset(train_batch_size)
        val_data = get_validation_dataset(valid_batch_size)
        test_data = get_test_dataset(test_batch_size)  

        train_data_len = count_data_items(TRAINING_FILENAMES)
        val_data_len = count_data_items(VALIDATION_FILENAMES)
        test_data_len = count_data_items(TEST_FILENAMES)      

        return train_data, val_data, test_data, train_data_len, val_data_len, test_data_len

    def save_data_as_image_file():
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
                        dirname = os.path.join(GCS_PATH, "images", title_list[did], label)            
                        os.makedirs(dirname, exist_ok=True)
                        filename = os.path.join(dirname, "%d_%d.jpg"%(i, imgid))
                    else:                    
                        label = "all"                
                        dirname = os.path.join(GCS_PATH, "images", title_list[did], label)            
                        os.makedirs(dirname, exist_ok=True)
                        filename = os.path.join(dirname, "%s.jpg"%labels[imgid].numpy().decode())                    
                    image = image.numpy()*255
                    image[image>255]=255
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    try:                    
                        cv2.imwrite(filename, image)
                    except:
                        print(filename)
                        break

    # save_data_as_image_file()
