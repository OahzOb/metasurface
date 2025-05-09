train_loader, val_loader, test_loader, param_size, spectra_size, dataname = data_preprocess.prepare_data_Full(batch_size = 1, transform = transform, subscale = False)
        PNNreal = my_model.PNN(input_size = param_size, output_size = spectra_size, origin = False).to(device)
        model_dict, time_stamp, epoch = my_model.get_model_dict(f'PNNreal_{dataname}', device)
        if model_dict == None:
            exit(f'There is no PNNreal_{dataname} ready for testing.')
        PNNreal.load_state_dict(model_dict)
        PNNimag = my_model.PNN(input_size = param_size, output_size = spectra_size, origin = False).to(device)
        model_dict, time_stamp, epoch = my_model.get_model_dict(f'PNNimag_{dataname}', device)
        if model_dict == None:
            exit(f'There is no PNNimag_{dataname} ready for testing.')
        PNNimag.load_state_dict(model_dict)
        model = my_model.Fullmodel(input_size = spectra_size, output_size = param_size, PNNreal = PNNreal, PNNimag = PNNimag, origin = False).to(device)
        prefix = f'Fullmodel_{dataname}'
        model_dict, time_stamp, epoch = my_model.get_model_dict(prefix, device)
        if model_dict is not None:
            model.load_state_dict(model_dict)
        print('Data Loaded!')
        print(model)
        losses = []
        errors = []
        errors_abs = []
        all_params = []
        all_targets = []
        all_outputs = []
        criterion = my_model.custom_criterion(min_val = 0, max_val = 0, device = device, dataname = dataname, mode = 'mean')
        mre = my_model.MRE()
        mare = my_model.MARE()
        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                params, outputs = model(inputs)
                loss = criterion((params, outputs), targets)
                error = mre(outputs, targets)
                error_abs = mare(outputs, targets)
                losses.append(loss.item())
                errors.append(error.item())
                errors_abs.append(error_abs.item())
                all_params.append(params.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
        losses = np.array(losses)
        errors = np.array(errors)
        errors_abs = np.array(errors_abs)
        all_params = np.concatenate(all_params, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)

        # 计算平均指标
        avg_loss = np.mean(losses)
        avg_error = np.mean(errors)
        avg_error_abs = np.mean(errors_abs)

        # 输出结果
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Error (MRE): {avg_error:.4f}")
        print(f"Average Absolute Error (MARE): {avg_error_abs:.4f}")