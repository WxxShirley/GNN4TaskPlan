import os 
import torch 


def save_checkpoint(model, optimizer, cur_epoch, args, is_best=False):
    os.makedirs(f"{args.output_dir}/{args.dataset}/{args.llm}", exist_ok=True)

    param_grad_dict = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }

    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dict and not param_grad_dict[k]:
            del state_dict[k]
    
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": args,
        "epoch": cur_epoch
    }

    path = f"{args.output_dir}/{args.dataset}/{args.llm}/{args.gnn_type}_Epoch{args.num_epochs}_checkpoint_{'best' if is_best else cur_epoch}.pth"
    print("Saving checkpoint at epoch {} to {}".format(cur_epoch, path))
    torch.save(save_obj, path)


def reload_best_model(model, args):
    checkpoint_path = f"{args.output_dir}/{args.dataset}/{args.llm}/{args.gnn_type}_Epoch{args.num_epochs}_checkpoint_best.pth"
    
    print("Loading checkpoint from {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model 


def reload_model(model, checkpoint_path):
    print("Loading checkpoint from {}".format(checkpoint_path))
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    
    return model 
