load(append(path,loc))
filename = erase(loc,".mat");
fullstruct = eval(filename);
sessiontable = table();
for name = ["target","upper","lower","startTime"]
    sessiontable.(name) = [fullstruct.(name)].';
end
temp =  {fullstruct.tap1Times};
j = 1;
sess_size = [];
sessnumbs = [];
for i=temp
    sess_size=[sess_size,length(i{1}(:,1))];
    sessnumbs=[sessnumbs,ones(1,length(i{1}(:,1)))*j];
    j=j+1;
end
presstable = table();
for name = ["tap1Times","tap2Times"]
    bothrows = [vertcat(fullstruct.(name))];
    presstable.(append(name,"_on")) = bothrows(:,1);
    presstable.(append(name,"_off")) = bothrows(:,2);
end
presstable.n_sess = sessnumbs.';
sessiontable.sess_size = sess_size.';
presstable.reward = vertcat(fullstruct.rewards);
writetable(sessiontable,append(path,append(filename,"_sessinfo",".csv")));
writetable(presstable,append(path,append(filename,"_pressinfo",".csv")));
