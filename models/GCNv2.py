import torch


class GCNv2(torch.nn.Module):
  def __init__(self):
    super(GCNv2, self).__init__()
    self.elu = torch.nn.ELU(inplace=True)

    self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
    self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)

    self.conv3_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.conv3_2 = torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)

    self.conv4_1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.conv4_2 = torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)

    # Descriptor
    self.convF_1 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.convF_2 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    # Detector
    self.convD_1 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.convD_2 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    self.pixel_shuffle = torch.nn.PixelShuffle(16)

  def forward(self, x):
    _, _, height, width = x.shape
    x = self.elu(self.conv1(x))
    x = self.elu(self.conv2(x))

    x = self.elu(self.conv3_1(x))
    x = self.elu(self.conv3_2(x))

    x = self.elu(self.conv4_1(x))
    x = self.elu(self.conv4_2(x))

    # Descriptor xF
    xF = self.elu(self.convF_1(x))
    desc = self.convF_2(xF)
    dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
    coarse_desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

    # Detector xD
    xD = self.elu(self.convD_1(x))
    det = self.convD_2(xD)
    semi = self.pixel_shuffle(det).squeeze()
    pts = torch.nonzero(semi >= 0.05, as_tuple=False).flip(-1).permute(1, 0)
    pts = torch.cat((pts.float(), torch.unsqueeze(semi[pts[1, :], pts[0, :]], dim=0)), dim=0)

    D = coarse_desc.size(1)
    samp_pts = torch.cat((torch.unsqueeze(torch.div(pts[0, :], width // 2).add(-1), 0),
                          torch.unsqueeze(torch.div(pts[1, :], height // 2).add(-1), 0)), 0)
    samp_pts = samp_pts.transpose(0, 1).contiguous()
    samp_pts = samp_pts.view(1, 1, -1, 2)
    samp_pts = samp_pts.float()
    desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
    desc = desc.reshape(D, -1)
    desc /= torch.unsqueeze(torch.norm(desc, dim=0), 0)
    desc = torch.clamp(desc.sign(), 0, 1)
    return pts, desc

if __name__ == "__main__":
    import os
    from torchstat import stat
    from thop import profile
    import time

    with torch.no_grad():
        model = GCNv2().cuda()
        model.eval()
        input = torch.rand([1, 1, 960, 1280]).cuda()
        output = model.forward(input)

        tic = time.time()
        for _ in range(100):
            output = model.forward(input)
        toc = time.time()
    print("Forward Flow Done, cost {} ms".format((toc - tic) * 1000 / 100))

    stat(model.cpu(), (1, 480, 640))