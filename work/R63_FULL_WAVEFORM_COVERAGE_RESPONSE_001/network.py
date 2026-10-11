"""Full-window stochastic displacement network; no paired waveform teacher."""
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__();self.n1=nn.GroupNorm(8,cin);self.c1=nn.Conv1d(cin,cout,3,padding=1)
        self.n2=nn.GroupNorm(8,cout);self.film=nn.Linear(128,2*cout);self.c2=nn.Conv1d(cout,cout,3,padding=1)
        self.skip=nn.Identity() if cin==cout else nn.Conv1d(cin,cout,1)
    def forward(self,x,e):
        z=self.c1(torch.nn.functional.silu(self.n1(x)));scale,shift=self.film(e).chunk(2,1)
        z=self.n2(z)*(1+scale[:,:,None])+shift[:,:,None]
        return self.skip(x)+self.c2(torch.nn.functional.silu(z))

class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(128);self.attn=nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
    def forward(self,x):
        z=x.transpose(1,2);n=self.norm(z)
        return (z+self.attn(n,n,n,need_weights=False)[0]).transpose(1,2)

class ResponseNetwork(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Sequential(nn.Linear(9,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.register_buffer('position',torch.linspace(-1.,1.,480).reshape(1,1,480))
        self.input=nn.Conv1d(3,32,3,padding=1)
        self.enc=nn.ModuleList([nn.ModuleList([ResidualBlock(c,c),ResidualBlock(c,c)]) for c in [32,64,128,128]])
        self.down=nn.ModuleList([nn.Conv1d(a,b,3,stride=2,padding=1) for a,b in [(32,64),(64,128),(128,128)]])
        self.attention=Attention()
        self.up=nn.ModuleList([nn.Conv1d(a,b,3,padding=1) for a,b in [(128,128),(128,64),(64,32)]])
        self.dec=nn.ModuleList([nn.ModuleList([ResidualBlock(2*c,c),ResidualBlock(c,c)]) for c in [128,64,32]])
        self.final_norm=nn.GroupNorm(8,32);self.final=nn.Conv1d(32,1,3,padding=1)
        nn.init.zeros_(self.final.weight);nn.init.zeros_(self.final.bias)
    def forward(self,noisy,b,guide,charge_condition):
        y=noisy.to(self.final.weight);g=guide.to(y);h=charge_condition.to(y).reshape(-1,1)
        onehot=torch.nn.functional.one_hot(b.to(torch.int64),8).to(y)
        e=self.embedding(torch.cat((h,onehot),1))
        z=self.input(torch.cat((y[:,None],g[:,None],self.position.to(y).expand(len(y),-1,-1)),1));skips=[]
        for j,blocks in enumerate(self.enc):
            z=blocks[0](z,e)
            if j==3:z=self.attention(z)
            z=blocks[1](z,e);skips.append(z)
            if j<3:z=self.down[j](z)
        for j,(up,blocks) in enumerate(zip(self.up,self.dec)):
            z=up(torch.nn.functional.interpolate(z,scale_factor=2,mode='nearest'))
            z=blocks[1](blocks[0](torch.cat((z,skips[2-j]),1),e),e)
        return self.final(torch.nn.functional.silu(self.final_norm(z)))[:,0]

# Stable name for the frozen evaluator/inference interface; this is a static field.
FlowNetwork=ResponseNetwork
