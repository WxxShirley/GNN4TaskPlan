DATASET=huggingface

# GCN
#  - LM Freeze # Namespace(batch_size=512, dataset='huggingface', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GCN', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=1, train_num=3000) 
#  - LM Tune  # Namespace(batch_size=512, dataset='huggingface', device='cuda:0', dynamic_sample=0, epoch=20, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GCN', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=1, train_num=3000) 
python main.py --lm_frozen=1 --epoch=10 --text_negative=1 --gnn_name=GCN --lr=0.001
python main.py --lm_frozen=0 --epoch=20 --text_negative=1 --gnn_name=GCN 
 

# GAT
#  - LM Freeze # Namespace(batch_size=512, dataset='huggingface', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GAT', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=0, train_num=3000) 
#  - LM Tune  # Namespace(batch_size=512, dataset='huggingface', device='cuda:0', dynamic_sample=0, epoch=20, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GAT', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=1, train_num=3000) 
python main.py --lm_frozen=1 --epoch=10 --text_negative=0 --gnn_name=GAT --lr=0.001
python main.py --lm_frozen=0 --epoch=20 --text_negative=1 --gnn_name=GAT


# SAGE
#  - LM Freeze # Namespace(batch_size=512, dataset='huggingface', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='SAGE', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=1, train_num=3000) 
#  - LM Tune  # Namespace(batch_size=512, dataset='huggingface', device='cuda:0', dynamic_sample=0, epoch=20, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='SAGE', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=1, train_num=3000) 
python main.py --lm_frozen=1 --epoch=10 --text_negative=1 --gnn_name=SAGE --lr=0.001
python main.py --lm_frozen=0 --epoch=20 --text_negative=1 --gnn_name=SAGE


# GIN
#  - LM Freeze # Namespace(batch_size=512, dataset='huggingface', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GIN', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=0, train_num=3000) 
#  - LM Tune  # Namespace(batch_size=512, dataset='huggingface', device='cuda:0', dynamic_sample=0, epoch=20, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GIN', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=1, train_num=3000) 
python main.py --lm_frozen=1 --epoch=10 --text_negative=0 --gnn_name=GIN --lr=0.001
python main.py --lm_frozen=0 --epoch=20 --text_negative=1 --gnn_name=GIN


# TransformerConv
#  - LM Freeze # Namespace(batch_size=512, dataset='huggingface', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='TransformerConv', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=1, train_num=3000) 
#  - LM Tune  # Namespace(batch_size=512, dataset='huggingface', device='cuda:0', dynamic_sample=0, epoch=20, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='TransformerConv', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=1, train_num=3000) 
python main.py --lm_frozen=1 --epoch=10 --text_negative=1 --gnn_name=TransformerConv --lr=0.001
python main.py --lm_frozen=0 --epoch=20 --text_negative=1 --gnn_name=TransformerConv
# (重新训练的时候co-train用的epoch=10) 
 
# SGC
#  - LM Freeze # Namespace(batch_size=512, dataset='huggingface', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='SGC', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=1, train_num=3000) 
#  - LM Tune  # Namespace(batch_size=512, dataset='huggingface', device='cuda:0', dynamic_sample=0, epoch=20, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='SGC', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=1, train_num=3000) 
python main.py --lm_frozen=1 --epoch=10 --text_negative=1 --gnn_name=SGC --lr=0.001
python main.py --lm_frozen=0 --epoch=20 --text_negative=1 --gnn_name=SGC
 
 


DATASET=dailylife

# GCN
#  - LM Freeze # Namespace(batch_size=512, dataset='dailylife', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GCN', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=1, train_num=3000) 
#  - LM Tune # Namespace(batch_size=512, dataset='dailylife', device='cuda:0', dynamic_sample=0, epoch=6, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GCN', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=0, train_num=3000) 
python main.py --dataset=$DATASET --lm_frozen=1 --epoch=10 --gnn_name=GCN --text_negative=1  --lr=0.001 
python main.py --dataset=$DATASET --lm_frozen=0 --epoch=6  --gnn_name=GCN --text_negative=0 


# GAT
#  - LM Freeze # Namespace(batch_size=512, dataset='dailylife', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GAT', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=1, train_num=3000) 
#  - LM Tune # Namespace(batch_size=512, dataset='dailylife', device='cuda:0', dynamic_sample=0, epoch=6, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GAT', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=0, train_num=3000) 
python main.py --dataset=$DATASET --lm_frozen=1 --epoch=10 --gnn_name=GAT --text_negative=1  --lr=0.001 
python main.py --dataset=$DATASET --lm_frozen=0 --epoch=6  --gnn_name=GAT --text_negative=0 


# SAGE
#  - LM Freeze # Namespace(batch_size=512, dataset='dailylife', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='SAGE', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=1, train_num=3000) 
#  - LM Tune # Namespace(batch_size=512, dataset='dailylife', device='cuda:0', dynamic_sample=0, epoch=6, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='SAGE', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=1, train_num=3000) 
python main.py --dataset=$DATASET --lm_frozen=1 --epoch=10 --gnn_name=SAGE --text_negative=1  --lr=0.001 
python main.py --dataset=$DATASET --lm_frozen=0 --epoch=6  --gnn_name=SAGE --text_negative=1 


# GIN
#  - LM Freeze # Namespace(batch_size=512, dataset='dailylife', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GIN', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=0, train_num=3000) 
#  - LM Tune # Namespace(batch_size=512, dataset='dailylife', device='cuda:0', dynamic_sample=0, epoch=6, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GIN', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=0, train_num=3000) 
python main.py --dataset=$DATASET --lm_frozen=1 --epoch=10 --gnn_name=GIN --text_negative=0  --lr=0.001 
python main.py --dataset=$DATASET --lm_frozen=0 --epoch=6  --gnn_name=GIN --text_negative=0 


# TransformerConv
#  - LM Freeze # Namespace(batch_size=512, dataset='dailylife', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='TransformerConv', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=1, train_num=3000) 
#  - LM Tune # Namespace(batch_size=128, dataset='dailylife', device='cuda:0', dynamic_sample=0, epoch=6, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='TransformerConv', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=0, train_num=3000) 
python main.py --dataset=$DATASET --lm_frozen=1 --epoch=10 --gnn_name=TransformerConv --text_negative=1  --lr=0.001 
python main.py --dataset=$DATASET --lm_frozen=0 --epoch=6  --gnn_name=TransformerConv --text_negative=0  --batch_size=128


# SGC
#  - LM Freeze # Namespace(batch_size=512, dataset='dailylife', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='SGC', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=0, train_num=3000) 
#  - LM Tune # Namespace(batch_size=512, dataset='dailylife', device='cuda:0', dynamic_sample=0, epoch=6, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='SGC', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=1, train_num=3000) 
python main.py --dataset=$DATASET --lm_frozen=1 --epoch=10 --gnn_name=SGC --text_negative=0  --lr=0.001 
python main.py --dataset=$DATASET --lm_frozen=0 --epoch=6  --gnn_name=SGC --text_negative=1  





DATASET=multimedia

# GCN 
#  - LM Freeze # Namespace(batch_size=512, dataset='multimedia', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GCN', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=1, train_num=3000) 
#  - LM Tune # Namespace(batch_size=512, dataset='multimedia', device='cuda:0', dynamic_sample=0, epoch=20, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GCN', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=1, train_num=3000) 
python main.py --dataset=$DATASET --epoch=10 --lm_frozen=1 --gnn_name=GCN --text_negative=1  --lr=1e-3 
python main.py --dataset=$DATASET --epoch=20 --lm_frozen=0 --gnn_name=GCN --text_negative=1 


# GAT
#  - LM Freeze # Namespace(batch_size=512, dataset='multimedia', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GAT', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=1, train_num=3000) 
#  - LM Tune # Namespace(batch_size=512, dataset='multimedia', device='cuda:0', dynamic_sample=0, epoch=20, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GAT', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=1, train_num=3000) 
python main.py --dataset=$DATASET --epoch=10 --lm_frozen=1 --gnn_name=GAT --text_negative=1  --lr=1e-3 
python main.py --dataset=$DATASET --epoch=20 --lm_frozen=0 --gnn_name=GAT --text_negative=1 


# SAGE
#  - LM Freeze # Namespace(batch_size=512, dataset='multimedia', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='SAGE', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=0, train_num=3000) 
#  - LM Tune # Namespace(batch_size=512, dataset='multimedia', device='cuda:0', dynamic_sample=0, epoch=20, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='SAGE', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=0, train_num=3000) 
python main.py --dataset=$DATASET --epoch=10 --lm_frozen=1 --gnn_name=SAGE --text_negative=0  --lr=1e-3 
python main.py --dataset=$DATASET --epoch=20 --lm_frozen=0 --gnn_name=SAGE --text_negative=0 


# GIN
#  - LM Freeze # Namespace(batch_size=512, dataset='multimedia', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GIN', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=0, train_num=3000) 
#  - lM Tune # Namespace(batch_size=512, dataset='multimedia', device='cuda:0', dynamic_sample=0, epoch=20, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='GIN', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=1, train_num=3000) 
python main.py --dataset=$DATASET --epoch=10 --lm_frozen=1 --gnn_name=GIN --text_negative=0  --lr=1e-3 
python main.py --dataset=$DATASET --epoch=20 --lm_frozen=0 --gnn_name=GIN --text_negative=1 


# TransformerConv
#  - LM Freeze # Namespace(batch_size=512, dataset='multimedia', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='TransformerConv', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=0, train_num=3000) 
#  - LM Tune # Namespace(batch_size=512, dataset='multimedia', device='cuda:0', dynamic_sample=0, epoch=20, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='TransformerConv', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=1, train_num=3000) 
python main.py --dataset=$DATASET --epoch=10 --lm_frozen=1 --gnn_name=TransformerConv --text_negative=0  --lr=1e-3 
python main.py --dataset=$DATASET --epoch=20 --lm_frozen=0 --gnn_name=TransformerConv --text_negative=1 


# SGC
#  - LM Freeze # Namespace(batch_size=512, dataset='multimedia', device='cuda:0', dynamic_sample=0, epoch=10, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='SGC', lm_frozen=1, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=0.001, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=1, seed=0, text_negative=1, train_num=3000) 
#  - LM Tune # Namespace(batch_size=512, dataset='multimedia', device='cuda:0', dynamic_sample=0, epoch=20, gnn_hidden_dim=1024, gnn_layer=1, gnn_name='SGC', lm_frozen=0, lm_name='intfloat/e5-large', load_alignment=True, load_model=0, lr=2e-05, maximum='', measure='dot', no_gnn_ablation=1, num_negatives=2, patience=5, save_model=0, seed=0, text_negative=1, train_num=3000) 
python main.py --dataset=$DATASET --epoch=10 --lm_frozen=1 --gnn_name=SGC --text_negative=1  --lr=1e-3 
python main.py --dataset=$DATASET --epoch=20 --lm_frozen=0 --gnn_name=SGC --text_negative=1 
