%% plot color pics
clear; clc;
load(['./simulation_results/results/','truth','.mat']);

load(['./simulation_results/results/','TwIST','.mat']);
pred_block_twist = pred;

load(['./simulation_results/results/','GAP-TV','.mat']);
pred_block_gaptv = pred;

load(['./simulation_results/results/','DeSCI','.mat']);
pred_block_desci = pred;

load(['./simulation_results/results/','tsa_net','.mat']);
pred_block_tsanet = pred;

load(['./simulation_results/results/','gap_net','.mat']);
pred_block_gapnet = pred;

load(['./simulation_results/results/','mst_l','.mat']);
pred_block_mst_l = pred;

load(['./simulation_results/results/','dauhst_9stg','.mat']);
pred_block_dauhst = pred;

load(['./simulation_results/results/','DERNN_LNLT_9stg','.mat']);
pred_block_dernn_lnlt_9stg = pred;

load(['./simulation_results/results/','DERNN_LNLT_9stg_plus','.mat']);
pred_block_dernn_lnlt_9stg_plus = pred;

lam28 = [453.5 457.5 462.0 466.0 471.5 476.5 481.5 487.0 492.5 498.0 504.0 510.0...
    516.0 522.5 529.5 536.5 544.0 551.5 558.5 567.5 575.5 584.5 594.5 604.0...
    614.5 625.0 636.5 648.0];

truth(find(truth>0.7))=0.7;
pred_block_twist(find(pred_block_twist>0.7))=0.7;
pred_block_gaptv(find(pred_block_gaptv>0.7))=0.7;
pred_block_desci(find(pred_block_desci>0.7))=0.7;
pred_block_tsanet(find(pred_block_tsanet>0.7))=0.7;
pred_block_gapnet(find(pred_block_gapnet>0.7))=0.7;
pred_block_mst_l(find(pred_block_mst_l>0.7))=0.7;
pred_block_dauhst(find(pred_block_dauhst>0.7))=0.7;
pred_block_dernn_lnlt_9stg(find(pred_block_dernn_lnlt_9stg>0.7))=0.7;
pred_block_dernn_lnlt_9stg_plus(find(pred_block_dernn_lnlt_9stg_plus>0.7))=0.7;


f = 8;

%% plot spectrum
figure(123);
[yx, rect2crop]=imcrop(sum(squeeze(truth(f, :, :, :)), 3), [40 50 40 40]);
rect2crop=round(rect2crop)
% close(123);
imshow(yx / 28)
figure; 

spec_mean_truth = mean(mean(squeeze(truth(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_twist = mean(mean(squeeze(pred_block_twist(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_gaptv = mean(mean(squeeze(pred_block_gaptv(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_desci = mean(mean(squeeze(pred_block_desci(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_tsanet = mean(mean(squeeze(pred_block_tsanet(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_gapnet = mean(mean(squeeze(pred_block_gapnet(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_mst_l = mean(mean(squeeze(pred_block_mst_l(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_dauhst = mean(mean(squeeze(pred_block_dauhst(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_dernn_lnlt_9stg = mean(mean(squeeze(pred_block_dernn_lnlt_9stg(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_dernn_lnlt_9stg_plus = mean(mean(squeeze(pred_block_dernn_lnlt_9stg_plus(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);


spec_mean_truth = spec_mean_truth./max(spec_mean_truth);
spec_mean_twist = spec_mean_twist./max(spec_mean_twist);
spec_mean_gaptv = spec_mean_gaptv./max(spec_mean_gaptv);
spec_mean_desci = spec_mean_desci./max(spec_mean_desci);
spec_mean_tsanet = spec_mean_tsanet./max(spec_mean_tsanet);
spec_mean_gapnet = spec_mean_gapnet./max(spec_mean_gapnet);
spec_mean_mst_l = spec_mean_mst_l./max(spec_mean_mst_l);
spec_mean_dauhst = spec_mean_dauhst./max(spec_mean_dauhst);
spec_mean_dernn_lnlt_9stg = spec_mean_dernn_lnlt_9stg./max(spec_mean_dernn_lnlt_9stg);
spec_mean_dernn_lnlt_9stg_plus = spec_mean_dernn_lnlt_9stg_plus./max(spec_mean_dernn_lnlt_9stg_plus);


corr_twist = roundn(corr(spec_mean_truth(:),spec_mean_twist(:)),-4);
corr_gaptv = roundn(corr(spec_mean_truth(:),spec_mean_gaptv(:)),-4);
corr_desci = roundn(corr(spec_mean_truth(:),spec_mean_desci(:)),-4);
corr_tsanet = roundn(corr(spec_mean_truth(:),spec_mean_tsanet(:)),-4);
corr_gapnet = roundn(corr(spec_mean_truth(:),spec_mean_gapnet(:)),-4);
corr_mst_l = roundn(corr(spec_mean_truth(:),spec_mean_mst_l(:)),-4);
corr_dauhst = roundn(corr(spec_mean_truth(:),spec_mean_dauhst(:)),-4);
corr_dernn_lnlt_9stg = roundn(corr(spec_mean_truth(:),spec_mean_dernn_lnlt_9stg(:)),-4);
corr_dernn_lnlt_9stg_plus = roundn(corr(spec_mean_truth(:),spec_mean_dernn_lnlt_9stg_plus(:)),-4);



X = lam28;

Y(1,:) = spec_mean_truth(:); 
Y(2,:) = spec_mean_twist(:); Corr(1)=corr_twist;
Y(3,:) = spec_mean_gaptv(:); Corr(2)=corr_gaptv;
Y(4,:) = spec_mean_desci(:); Corr(3)=corr_desci;
Y(5,:) = spec_mean_tsanet(:); Corr(4)=corr_tsanet;
Y(6,:) = spec_mean_gapnet(:); Corr(5)=corr_gapnet;
Y(7,:) = spec_mean_mst_l(:); Corr(6)=corr_mst_l;
Y(8,:) = spec_mean_dauhst(:); Corr(7)=corr_dauhst;
Y(9,:) = spec_mean_dernn_lnlt_9stg(:); Corr(8)=corr_dernn_lnlt_9stg;
Y(10,:) = spec_mean_dernn_lnlt_9stg_plus(:); Corr(9)=corr_dernn_lnlt_9stg_plus;



createfigure(X,Y,Corr)


