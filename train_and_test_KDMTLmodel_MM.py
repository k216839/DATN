
def train_and_test_KDMTLmodel_MM(model, MultiMNISt_params):

    train_loader, val_loader, test_loader = load_MultiMnist_data()
    
    size_data_for_search = int(0.2 * len(train_loader.dataset))
    _ = len(train_loader.dataset) - size_data_for_search
    dataset_for_search, _ = torch.utils.data.random_split(train_loader.dataset, [size_data_for_search, _])
    dataloader_for_search = torch.utils.data.DataLoader(dataset_for_search, batch_size=256, shuffle=True)
    
    device = MultiMNISt_params["device"]
    print(f"Training... [-- running on {device} --]")
    
    MultiMNISt_params["data_search"] = dataloader_for_search
    
    final_model, ALL_TRAIN_LOSS, ALL_VAL_ACCU, MODEL_VAL_ACCU = full_training_kdmtl(train_loader, val_loader, model,
                          MultiMNISt_params, init_model = True)
    
    print("Training completed !") 

    print("Testing ...") 
    Test_accuracy, prec_wrong_images = test_multitask_kdmtl_model(test_loader, final_model, MultiMNISt_params, load = True) 

    return Test_accuracy, prec_wrong_images, ALL_TRAIN_LOSS, ALL_VAL_ACCU, MODEL_VAL_ACCU