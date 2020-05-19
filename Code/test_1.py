# # 归一化数据（均没有测试集）
# # mean = X_data.mean(axis=0)
# # X_data -= mean
# # std = X_data.std(axis=0)
# # X_data /= std
#
# print("sda is", X_data)
#
# def build_model():
#     '''
#     由于后面我们需要反复构造同一种结构的网络，所以我们把网络的构造代码放在一个函数中，
#     后面只要直接调用该函数就可以将网络迅速初始化
#     '''
#     model = models.Sequential()
#     model.add(layers.Dense(units=1, input_dim=18, activation='softmax'))
#     # model.add(layers.Dense(64, activation='relu', input_shape=(X_data.shape[1],)))
#     # model.add(layers.Dense(64, activation='relu'))
#     # model.add(layers.Dense(1))
#     # model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
#     model.compile(optimizer='SGD', loss='mse', metrics=['accuracy'])
#     return model
#
#
# # 切分数据集，训练
# k = 4
# num_val_samples = len(X_data) // k #整数除法
# num_epochs = 1
# all_mae_histories = []
#
# # for i in range(k):
# print('processing fold #', 1)
# #依次把k分数据中的每一份作为校验数据集
# val_data = X_data[1 * num_val_samples : (1+1) * num_val_samples]
# val_targets = Y_data[1 * num_val_samples : (1+1) * num_val_samples]
#
# #把剩下的k-1分数据作为训练数据,如果第i分数据作为校验数据，那么把前i-1份和第i份之后的数据连起来
# partial_train_data = np.concatenate([X_data[: 1 * num_val_samples],
#                                      X_data[(1+1) * num_val_samples:]], axis = 0)
# partial_train_targets = np.concatenate([Y_data[: 1 * num_val_samples],
#                                         Y_data[(1+1) * num_val_samples: ]],
#                                       axis = 0)
# print("build model")
# model = build_model()
#
# print(partial_train_data.shape)
# print(partial_train_targets.shape)
#
# #把分割好的训练数据和校验数据输入网络
# history = model.fit(partial_train_data, partial_train_targets,
#           # validation_data=(val_data, val_targets),
#           epochs = num_epochs,
#           batch_size = 1, verbose = 0)
# print('kkk')
# mae_history = history.history['val_mean_absolute_error']
# all_mae_histories.append(mae_history)
# print('kkk')
#
#
# average_mae_history = [
#     np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
# ]
#
# import matplotlib.pyplot as plt
# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()
