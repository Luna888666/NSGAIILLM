classdef NSGAIILLM2Q < ALGORITHM
% <multi> <real/integer> <constrained/none>
% NSGA-II with adaptive rotation based simulated binary crossover

%------------------------------- Reference --------------------------------
% L. Pan, W. Xu, L. Li, C. He, and R. Cheng, Adaptive simulated binary
% crossover for rotated multi-objective optimization, Swarm and
% Evolutionary Computation, 2021, 60: 100759.
%--------------------------------------------------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            Population = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);
            B  = eye(Problem.D);
            m  = 0.5*(Problem.upper - Problem.lower);
            ps = 0.5;
            %% 压缩数量，例子个数
            hiddenSize = 1;
            par_num=5;
            yu=10000;
            scores=[yu];
            scores=repmat(scores,1,par_num);
            objs=[];
            ya_objs=[];
            %% 数据初始化
            cishu=0;
            zong=0;
            flag=0;
            popu_size=size(Population);
            par_size=size(Population.decs)
            mA=ones(popu_size(2),1);
            mB=zeros(popu_size(2),1);
            %% 50
            num_mA=par_size(1);
            %% 30
            num=par_size(2);
            par_size_obj=size(Population.objs)
            %% 目标数量
            obj_num=par_size_obj(2);
            interval=9;
            xun=4;
            %% 0:ns 1:llm
            flag=0;
            pre=0;
            now=1;
            theta=0.1;
            %% Optimization
            while Algorithm.NotTerminated(Population)
                zong=zong+1;
                if zong<=10
                    flag=0;
                else
                    if mod(zong,4)==0
                        flag=0;
                    else
                        if pre==now || pre>now || now-pre<theta
                            flag=0;
                        else
                            flag=1;
                        end
                    end
                end
               
                if flag==0 
                    MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                    Offspring  = Reproduction(Problem,Population(MatingPool),{B,m,ps});
                    [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
                    [B,m,ps,Population] = UpdateParameter(Problem,Population);
                    pre=0;
                    now=1;
                    ns_igd=IGD(Population,Problem.optimum);
                else
                    MatingPool=TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                    half_MatingPool=MatingPool(1:num_mA/2);
                    %% type选择是替换交叉算子，旋转算子还是全部
                    %% 1：全部 2:交叉算子 3:旋转算子
                    type=2;
                    [good_points,max,repeat,ex_num]=findDuplicates(MatingPool,par_num,num_mA,type);
                    while length(repeat)~=par_num
                        [good_points,max,repeat,ex_num]=findDuplicates(MatingPool,par_num,num_mA,type);
                    end
                    excellent=Population(max);
                    %% 例子的决策向量和目标向量
                    objs=excellent.objs;
                    history=excellent.decs;
    
                    file_name="llm2.py";
                    temp=[];
                    [tempA,tempB,tempC,flag]= pyrunfile(file_name,["tempA","tempB","tempC","flag"],objs=objs,flag=flag,num=num,num_mA=num_mA,par_num=par_num,history=history,obj_num=obj_num,ex_num=ex_num);
                    tempA=double(tempA);
                    tempB=double(tempB);
                    tempC=double(tempC);
                    while flag~=0 || length(tempA)~=num || length(tempB)~=num || length(tempC)~=num 
                        cishu=cishu+1;
                        tempA=[];
                        tempB=[];
                        tempC=[];
                        [tempA,tempB,tempC,flag]= pyrunfile(file_name,["tempA","tempB","tempC","flag"],objs=objs,flag=flag,num=num,num_mA=num_mA,par_num=par_num,history=history,obj_num=obj_num,ex_num=ex_num);
                        tempA=double(tempA);
                        tempB=double(tempB);
                        tempC=double(tempC);
                    end
                    
                    tempA=double(tempA);
                    tempB=double(tempB);
                    tempC=double(tempC);
                    temp=[temp,tempA,tempB,tempC];
                    
                    
                    %% 合并要进行变异的种群
                    yu_Population=Population(MatingPool).decs;
                    %% randomIntegers = randperm(length(repeat), 2);
                    for i=1:3
                        %% ti_index=randomIntegers(i);
                        x=repeat(par_num-i+1);
                        zhi=MatingPool(x);
                        indexs=find(half_MatingPool==zhi);
                        for j=1:length(indexs)
                            x=indexs(j);
                            %if x<=50
                            yu_Population(x,:)=temp(:,(i-1)*num+1:i*num);
                            %end
                        end

                        % if i<3
                        %     y=good_points(i);
                        %     yu_Population(y,:)=temp(:,(i-1)*num+1:i*num);
                        % end
                       
                        
                        % if zong<80
                        %    y=good_points(i);
                        %    zhi=MatingPool(y);
                        %    indexs=find(half_MatingPool==zhi);
                        %    for j=1:length(indexs)
                        %        x=indexs(j);
                        %        yu_Population(x,:)=temp(:,(i-1)*num+1:i*num);
                        %    end
                        % end
                       
                    end

                    pre=-Spread2(CrowdDis)+mean(FrontNo);
                    Offspring  = Reproduction_LLM(Problem,yu_Population,{B,m,ps});
                    [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
                    [B,m,ps,Population] = UpdateParameter(Problem,Population);
                    nsllm_igd=IGD(Population,Problem.optimum);
                    now=-Spread2(CrowdDis)+mean(FrontNo);
                    disp(zong);
                % else
                %    MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                %     test1=Population(MatingPool);
                %     Offspring  = OperatorGA(Problem,Population(MatingPool));
                %     [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
                %     nsllm_igd=IGD(Population,Problem.optimum);
                %     disp(zong);
                end
                
            end
        end
    end
end