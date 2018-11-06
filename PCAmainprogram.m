clc
clear all
close all

%%%%%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%
imlist=dir('./enroll/LargeDataSet/enrolling/*.bmp');
im =imread(['./enroll/LargeDataSet/enrolling/',imlist(1).name]);
[r,c]=size(im);
num_im=length(imlist);
num_p=num_im/5;
x=zeros(r*c,num_p);
im_vec=zeros(r*c,num_im);
Mec=zeros(r*c,1);
index=zeros;index2=zeros;
match=zeros(1,45);match2=zeros(1,45);
cmc=zeros(1,30);
cmc2=zeros(1,30);

%%%%%% convert all images to vector %%%%%%
for i=1:num_im
im =imread(['./enroll/LargeDataSet/enrolling/',imlist(i).name]);
%im = histeq(im); % improving training images
im_vec(:,i)=reshape(im',r*c,1);
end

%%%%%%%%%%%%%% to get xi and Me%%%%%%%%%%%%%%%%
j=1;
for i=1:5:(num_im-4)
x(:,j)=(im_vec(:,i)+im_vec(:,i+1)+im_vec(:,i+2)+im_vec(:,i+3)+im_vec(:,i+4))./5;
Mec(:,1)=Mec(:,1)+im_vec(:,i)+im_vec(:,i+1)+im_vec(:,i+2)+im_vec(:,i+3)+im_vec(:,i+4);
j=j+1;
end
Me=Mec(:,1)./num_im;

%%%%%%%%%%%%%% to get big A %%%%%%%%%%%%%%%%%%%%
for i=1:num_p
a(:,i)=x(:,i)-Me;
end

%%%%%%%%%%%%%% to get eig of A'*A (P2) %%%%%%%%%
ata = a'*a;  
[V,D] = eig(ata);
p2 = [];
for i = 1 : size(V,2) 
    if( D(i,i)>1 )
        p2 = [p2 V(:,i)];
    end
end
%%%weight of the training data projected into eigen space%%%%%
wta=p2'*ata;
%% Plot the weight for person 1
for n = 1:1
    figure,plot(wta(:,n));
end

%%%%%%%%%%%%%% to get the Eigenfaces %%%%%%%%%%%%
ef =a*p2;  %here is P you need to use in matching 
[rr,cc]=size(ef);

%%show the eigenface images ...
for n = 1:cc
    eigim_t = ef(:,n);
    eigface(:,:,n) = reshape(eigim_t,r,c);
    figure,imagesc(eigface(:,:,n)');
    axis image;axis off;colormap(gray(256));
    title('Eigen Face Image','fontsize',10);
    %imwrite(eigface(:,:,n),strcat('./Images/output/eig_face',num2str(n),'.bmp'));
end

%% %%%%%%%%%%%%%%%%%%%%%  TESTING  %%%%%%%%%%%%%%%%%%%%%%%%
imlist2=dir('./enroll/LargeDataSet/testing/*.bmp');
num_imt=length(imlist2);
imt_vector=zeros(r*c,num_imt);

%%%%%% convert all test images to vector %%%%%%
for i=1:num_imt
im =imread(['./enroll/LargeDataSet/testing/',imlist2(i).name]);
%im = histeq(im); % trying to improve the testing image
imt_vector(:,i)=reshape(im',r*c,1);
%%%%% get B=y-me %%%%%%%
   b(:,i)=imt_vector(:,i)-Me;
   wtb=ef'*b(:,i);
for ii=1:num_p   %% weight compare wtb and wta(i)
    eud(ii) = sqrt(sum((wtb-wta(:,ii)).^2));
end

[cdata,index(i)] = min(eud); %%find minimum eud's index

%%%%%%%%%%%%%%%%%%%%%%%  RESULT  %%%%%%%%%%%%%%%%%%%%%%%%
%%% right result by observation %%%
result = [1 1 2 3 3 4 5 5 5 6 7 7 8 8 9 9 10 10 13 13 16 16 17 18 19 ...
          20 21 22 23 25 27 29 29 30 31 32 33 34 36 38];
%%%%%%%%%%%%%%% CMC calculation (compare with result)%%%%%%%
if index(i) == result(i)
    match(1) = match(1)+1;
else
    [svals,idx] = sort(eud(:));
    index2(i) = idx(2);
    
    for y = 2:num_imt-1
        if index2(i) == result(i)
            index(i) = index2(i);
            match(y) = match(y)+1;
            break
        else
            index2(i) = idx(y+1);
        end
    end
end

end

%%%%%%%%%%%%%%% CMC curve plot %%%%%%%
 
 for i = 1:30
     cmc(:,i) = sum(match(1:i)/num_imt);
 end
 
 figure,plot(cmc);