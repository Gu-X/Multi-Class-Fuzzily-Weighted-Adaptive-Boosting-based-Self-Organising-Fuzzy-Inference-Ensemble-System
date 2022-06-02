function [PredFAS,AccFAS]=FWAdaBoostSOFIES(dataname,NumBaseLearner,GranLevel)
load (dataname)
TraData=DTra1;
TraLabel=LTra1;
TesData=DTes1;
TesLabel=LTes1;
UNL=unique(TraLabel);
CL=length(UNL);
%%
alpha=[];
DistanceType='Cosine';
Output1={};
RESU0=0;
[L,W]=size(TraData);
FinalTesLabel=zeros(length(TesLabel),CL);
Weights=ones(L,1)/L;
for ii=1:1:NumBaseLearner
    seq1=cumsum(Weights);
    seq=zeros(1,L);
    if ii~=1
        rand_num=rand(1,L);
        for i=1:L
            loc=find(rand_num(i)>seq1,1,'last');
            if isempty(loc)
                seq(i)=1;
            else
                seq(i)=loc;
            end
        end
    else
        seq=1:1:L;
    end
    seq=unique(seq);
    TraData1=TraData(seq,:);
    TraLabel1=TraLabel(seq);
    %%
    Input.TrainingData=TraData1;
    Input.TrainingLabel=TraLabel1;
    [Output1{ii}]=SOFClassifier(Input,GranLevel,'OfflineTraining',DistanceType);
    %%
    Input=Output1{ii};
    Input.TestingData=TraData;
    Input.TestingLabel=TraLabel;
    temp=full(ind2vec(TraLabel',CL))';
    [Output2]=SOFClassifier(Input,GranLevel,'Validation',DistanceType);
    PreTraLabel=Output2.EstimatedLabel;
    PreTraSoC=zeros(length(PreTraLabel),CL);
    PreTraSoC(:,unique(TraLabel1))=exp(-1*Output2.SoC);
    PreTraSoC1=max(PreTraSoC.*temp,[],2);
    PreTraSoC2=max(PreTraSoC.*(1-temp),[],2);
    Seq=(PreTraSoC1-PreTraSoC2);
    Input=Output1{ii};
    Input.TestingData=TesData;
    Input.TestingLabel=TesLabel;
    [Output2]=SOFClassifier(Input,GranLevel,'Validation',DistanceType);
    PreTesLabel=Output2.EstimatedLabel;
    PreTesSoC=zeros(length(PreTesLabel),CL);
    PreTesSoC(:,Input.TrainedClassifier.seq)=exp(-1*Output2.SoC);
    PreTesSoC=sort(PreTesSoC,2,'descend');
    Seq1=(PreTesSoC(:,1)-PreTesSoC(:,2));
    PreTesLabel1=(full(ind2vec(PreTesLabel',CL)))';
    PreTesLabel1(PreTesLabel1==0)=-1/(CL-1);
    %%
    seq1=PreTraLabel~=TraLabel;
    errorrate=sum(seq1.*Weights);alpha(ii)=0;
    if errorrate<=(CL-1)/CL&& errorrate>0
        alpha(ii)=log((1/errorrate)-1)+log(CL-1);
        FinalTesLabel=FinalTesLabel+alpha(ii)*PreTesLabel1.*repmat(Seq1,1,CL);
        Weights=Weights.*exp(-0.5*alpha(ii)*Seq);
        Weights=Weights/sum(Weights);
    end
end
[~,PredFAS]=max(FinalTesLabel,[],2);
CM=confusionmat(TesLabel,PredFAS);
AccFAS=sum(sum(CM.*eye(length(CM(1,:)))))./sum(sum(CM));
end
function [Output]=SOFClassifier(Input,GranLevel,Mode,DistanceType)
if strcmp(Mode,'OfflineTraining')==1
    data_train=Input.TrainingData;
    label_train=Input.TrainingLabel;
    seq=unique(label_train);
    data_train1={};
    N=length(seq);
    %%
    if strcmp(DistanceType,'Cosine')==1
        data_train=data_train./(repmat(sqrt(sum(data_train.^2,2)),1,size(data_train,2)));
        DistanceType='Euclidean';
    end
    if strcmp(DistanceType,'Euclidean')==1
        AXX=mean(sum(data_train.^2,2),1)-sum((mean(data_train,1)).^2);
        for i=1:1:N
            data_train1{i}=data_train(label_train==seq(i),:);
            delta(i)=mean(sum(data_train1{i}.^2,2))-sum(mean(data_train1{i},1).^2);
            [centre{i},Member{i},averdist{i}]=offline_training_Euclidean(data_train1{i},delta(i),GranLevel);
        end
        L=zeros(1,N);
        mu={};
        XX=zeros(1,N);
        ratio=zeros(1,N);
        for i=1:1:N
            mu{i}=mean(data_train1{i},1);
            [L(i),W]=size(data_train1{i});
            XX(i)=0;
            for ii=1:1:L(i)
                XX(i)=XX(i)+sum(data_train1{i}(ii,:).^2);
            end
            XX(i)=XX(i)./L(i);
            ratio(i)=averdist{i}/(2*(XX(i)-sum(mu{i}.^2)));
        end
        TrainedClassifier.seq=seq;
        TrainedClassifier.ratio=ratio;
        TrainedClassifier.miu=mu;
        TrainedClassifier.XX=XX;
        TrainedClassifier.L=L;
        TrainedClassifier.centre=centre;
        TrainedClassifier.Member=Member;
        TrainedClassifier.averdist=mean(delta);
        TrainedClassifier.NoC=N;
        TrainedClassifier.delta=delta;
    end
    Output.TrainedClassifier=TrainedClassifier;
end
%%
if strcmp(Mode,'Validation')==1
    TrainedClassifier=Input.TrainedClassifier;
    seq=TrainedClassifier.seq;
    data_test=Input.TestingData;
    label_test=Input.TestingLabel;
    N=TrainedClassifier.NoC;
    if strcmp(DistanceType,'Cosine')==1
        data_test=data_test./(repmat(sqrt(sum(data_test.^2,2)),1,size(data_test,2)));
        DistanceType='Euclidean';
    end
    if strcmp(DistanceType,'Euclidean')==1
        centre=TrainedClassifier.centre;
        dist=zeros(size(data_test,1),N);
        for i=1:1:N
            dist(:,i)=min(pdist2(data_test,centre{i},'euclidean'),[],2).^2;
        end
        [~,label_est]=min(dist,[],2);
        label_est=seq(label_est);
    end
    Output.TrainedClassifier=Input.TrainedClassifier;
    Output.ConfusionMatrix=confusionmat(label_test,label_est);
    Output.EstimatedLabel=label_est;
    Output.SoC=dist;
end
end
function [centre3,Mnumber,averdist]=offline_training_Euclidean(data,delta,GranLevel)
[L,W]=size(data);
dist00=pdist(data,'euclidean').^2;
dist0=squareform(dist00);
dist00=sort(dist00,'ascend');
for i=1:GranLevel
    dist00(dist00>mean(dist00))=[];
end
averdist=mean(dist00);
[UD,J,K]=unique(data,'rows');
F = histc(K,1:numel(J));
LU=length(UD(:,1));
if LU>2
    %%
    density=sum(dist0,2)./sum(sum(dist0,2));
    density=F./density(J);
    dist=dist0(J,J);
    [~,pos]=max(density);
    seq=1:1:LU;
    seq=seq(seq~=pos);
    Rank=zeros(LU,1);
    Rank(1,:)=pos;
    for i=2:1:LU
        [aa,pos0]=min(dist(pos,seq));
        pos=seq(pos0);
        Rank(i,:)=pos;
        seq=seq(seq~=pos);
    end
    data2=UD(Rank,:);
    data2den=density(Rank);
    Gradient=zeros(2,LU-2);
    Gradient(1,:)=data2den(1:1:LU-2)-data2den(2:1:LU-1);
    Gradient(2,:)=data2den(2:1:LU-1)-data2den(3:1:LU);
    seq2=2:1:LU-1;
    seq1=find(Gradient(1,:)<0&Gradient(2,:)>0);
    if Gradient(2,LU-2)<0
        seq3=[1,seq2(seq1),LU];
    else
        seq3=[1,seq2(seq1)];
    end
    centre0=data2(seq3,:);
else
    centre0=data;
end
nc=size(centre0,1);
dist3=pdist2(centre0,data,'euclidean').^2;
[~,seq4]=min(dist3,[],1);
centre1=zeros(nc,W);
Mnumber=zeros(nc,1);
miu=mean(data,1);
cenden=zeros(1,nc);
for i=1:1:nc
    seq5=find(seq4==i);
    Mnumber(i)=length(seq5);
    centre1(i,:)=mean(data(seq5,:),1);
    cenden(i)= Mnumber(i)/(1+sum((centre1(i,:)-miu).^2)/delta);
end
if nc==1
    centre2=centre1;
else
    dist4=pdist(centre1,'euclidean').^2;
    dist5=squareform(dist4);
    seqme2=zeros(nc);
    seqme2(dist5<=averdist)=1;
    cendenmex=seqme2.*(repmat(cenden,nc,1));
    seq6=find(abs(max(cendenmex,[],2)-cenden')==0);
    centre2=centre1(seq6,:);
end
nc=size(centre2,1);
dist6=pdist2(centre2,data,'euclidean').^2;
[~,seq7]=min(dist6,[],1);
centre3=zeros(nc,W);
Mnumber=zeros(nc,1);
for i=1:1:nc
    seq8=find(seq7==i);
    Mnumber(i)=length(seq8);
    centre3(i,:)=mean(data(seq8,:),1);
end
end
