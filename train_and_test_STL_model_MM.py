
def train_and_test_STL_model_MM(model, MultiMNISt_params_STL, train_func = None, test_func = None):

    # Start timer
    import datetime
    from time import time
    print(datetime.datetime.now())
    t0 = time()

    train_loader, val_loader, test_loader = load_MultiMnist_data()
    
    print("Training ...") 
    
    if train_func is None:
        train_func = train_single_model
    
    final_model, ALL_TRAIN_LOSS, ALL_VAL_ACCU = train_func(train_loader, val_loader, model, MultiMNISt_params_STL)

    print("Training completed !") 
    

    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    print("Testing ...") 
    
    if test_func is None:
        test_func = test_single_model
        
    Test_accuracy, prec_wrong_images = test_func(test_loader, final_model, MultiMNISt_params_STL) 

    return Test_accuracy, prec_wrong_images, ALL_TRAIN_LOSS, ALL_VAL_ACCU
