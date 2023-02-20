import torch
from engine import train_one_epoch, evaluate
from model_load import get_model
from utils import optim_set
from dataloader import get_dataloader
from detector_args import DetectorArgs


if __name__ == "__main__":
    '''
    step0_dataprepare를 진행하게 되면 coco format(json)이 생성되며 이 데이터를 참조하여 dataloader, model train을 진행한다.
    detector_args를 수정하여 할당 gpu, model 선택, hyperparameter 셋팅, 학습 결과 interval 설정
    '''

    args = DetectorArgs(mode='train')
    device = torch.device('cuda:{}'.format(args.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(model_type=args.type, device=device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer, lr_scheduler = optim_set(args, parameters=params)
    train_loader, valid_loader = get_dataloader(args)

    for epoch in range(args.epochs):
        # evaluate(model, valid_loader, device)
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=args.print_freq)
        lr_scheduler.step()
        evaluate(model, valid_loader, device)
        if epoch % args.save_interval==0:
            print("save")
            torch.save(model.state_dict(), 'detector/models/{}_weight_pad{}.pth'.format(args.type, epoch))
        
    # Save model weights after training
