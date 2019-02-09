function data=removeNAN(dataold)

info=sum(dataold,2);
datanew=[];
for i=1:length(info)
    if isequaln(info(i),NaN)
        datanew=[datanew];
    else
        datanew=[datanew; dataold(i,:)];
    end
end
data=datanew;