from models.SuperPointNetOpen import *

def save_checkpoint(save_path, net_state, epoch, filename='checkpoint.pth.tar'):
    file_prefix = ['superPointNetOpen']
    # torch.save(net_state, save_path)
    filename = '{}_{}_{}'.format(file_prefix[0], str(epoch), filename)
    torch.save(net_state, save_path+filename)
    print("save checkpoint to ", filename)
    pass

model_state_dict = torch.load('weights/superpoint_v1.pth')

save_checkpoint(
    "/home/sheli/PycharmProjects/pytorch-superpoint/weights/",
    {
        "n_iter": 0,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": None,
        "loss": None,
    },
   0,
)