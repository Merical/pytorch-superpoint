import torch

def sos_loss(x, label):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1).item() # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    xa = x[:, label.data==-1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0) # D * (B * num_neg)
    xp = x[:, label.data==1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    xn = x[:, label.data==0]

    dist_an = torch.sum(torch.pow(xa - xn, 2), dim=0)
    dist_pn = torch.sum(torch.pow(xp - xn, 2), dim=0)

    return torch.sum(torch.pow(dist_an - dist_pn, 2)) ** 0.5 / nq

if __name__ == "__main__":
    descriptors_a = torch.rand([1000, 256]).cuda()
    descriptors_b = torch.rand([1000, 256]).cuda()
    # dist0 = descriptors_a.dot(descriptors_b)
    dist0 = descriptors_a.mm(descriptors_a.T)
    dist1 = torch.mm(descriptors_a.T, descriptors_b)
    dist2 = descriptors_a * descriptors_a
    dist3 = descriptors_a * descriptors_a
    dist0_1 = dist0.diag(torch.zeros([dist0.shape[0]]))
    print("Done")
