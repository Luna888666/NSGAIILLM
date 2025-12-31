"""
This script is an example of using the OpenAI API to create various interactions with a ChatGLM3 model.
It includes functions to:

1. Conduct a basic chat session, asking about weather conditions in multiple cities.
2. Initiate a simple chat in Chinese, asking the model to tell a short story.
3. Retrieve and print embeddings for a given text input.

Each function demonstrates a different aspect of the API's capabilities, showcasing how to make requests
and handle responses.
"""
import openai
import os
from openai import OpenAI
import re
import random
import numpy as np
import requests
import json
import sys
from numpy import random,mat
from http import HTTPStatus
import http.client
import math
from dashscope import Generation
import dashscope
from zhipuai import ZhipuAI
dashscope.api_key="sk-256d54efc7544a6480f0c2de60fb99c3"

# 0:实验室智谱 1:实验室通义千问 2:阿里云平台的智谱模型chatglm3-6b
# 3:清华智谱
# 4:chatgpt-3.5-turbo--聊天
# 5:chatgpt-3.5-turbo--内容补全
# 6:deepseek-r1接口
model=6
# 1:论文中的提示 2:智谱推荐的提示
prompt_type=1


base_url = "http://10.14.10.101:8000/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)

url = "http://10.132.219.97:6006/chat/"

def experiment_qwen(ques):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key="sk-bc44eb4faa7847169b407f04b5ffefbe",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system","content": "You are an expert in multi-objective optimization algorithms. Your task is to generate and improve solutions for given problems in this domain. You will analyze the given solutions, consider multiple objectives, and generate improved solutions by balancing trade-offs between objectives.Your expertise includes but is not limited to:- Understanding and applying various multi-objective optimization techniques such as NSGA-II, MOEA/D, SPEA2, etc.- Evaluating solutions using common metrics like Pareto front coverage, Inverted Generational Distance (IGD), Hypervolume (HV), and others.- Ensuring diversity and convergence in the solution sets.- Applying statistical and machine learning methods to enhance optimization processes.You will:1. Analyze provided solution sets and evaluate their performance using standard multi-objective optimization metrics.2. Generate new and improved solutions by suggesting modifications or completely new approaches.3. Provide detailed explanations for your suggestions, including the expected improvements in specific metrics.4. Ensure the solutions maintain a balance between different objectives, improving overall performance.5. Use your knowledge of multi-objective optimization literature and practices to guide your suggestions.When providing solutions, consider the following guidelines:- Prioritize the optimization of given objectives without compromising the diversity of the solution set.- Explain how the new solutions can improve metrics such as IGD, HV, and Pareto front coverage.- Provide insights on potential trade-offs and why certain solutions may perform better in specific scenarios."},
            {"role": "user", "content": ques}
    
        ]
    )
    data = json.loads(completion.model_dump_json())
    content = data["choices"][0]["message"]["content"]
    print(content)
    return content

def gpt_35(ques):
    messages = [{"role": "system","content": "You are an expert in multi-objective optimization algorithms. Your task is to generate and improve solutions for given problems in this domain. You will analyze the given solutions, consider multiple objectives, and generate improved solutions by balancing trade-offs between objectives.Your expertise includes but is not limited to:- Understanding and applying various multi-objective optimization techniques such as NSGA-II, MOEA/D, SPEA2, etc.- Evaluating solutions using common metrics like Pareto front coverage, Inverted Generational Distance (IGD), Hypervolume (HV), and others.- Ensuring diversity and convergence in the solution sets.- Applying statistical and machine learning methods to enhance optimization processes.You will:1. Analyze provided solution sets and evaluate their performance using standard multi-objective optimization metrics.2. Generate new and improved solutions by suggesting modifications or completely new approaches.3. Provide detailed explanations for your suggestions, including the expected improvements in specific metrics.4. Ensure the solutions maintain a balance between different objectives, improving overall performance.5. Use your knowledge of multi-objective optimization literature and practices to guide your suggestions.When providing solutions, consider the following guidelines:- Prioritize the optimization of given objectives without compromising the diversity of the solution set.- Explain how the new solutions can improve metrics such as IGD, HV, and Pareto front coverage.- Provide insights on potential trade-offs and why certain solutions may perform better in specific scenarios."},{'role': 'user', 'content': ques}]
    ##{"role": "system","content": "You are an expert at multi-objective optimization. You are proficient in the methods of updating population in various multi-objective optimization algorithms, including NSGAII, NSGAIII,MOEAD, NSGAII-ARSBX, and related variants."}
    headers = {
       'Authorization': 'Bearer fk227060-0ffcEWMqcVWtEHwNQP9kMm29U1OOQcGR',
       'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
       'Content-Type': 'application/json'
        }
    data = {"model": "gpt-3.5-turbo", "messages": messages,"safe_mode": False}
    # http://oa.api2d.net/v1/chat/completions--聊天
    # http://oa.api2d.net/v1/completions--内容补全
    try:
        response = requests.post(url='http://oa.api2d.net/v1/chat/completions', headers=headers, data=json.dumps(data))
    except:
        return ""
        print("连接错误")
    # print(response.json()['choices'][0]['message']['content'])
    try:
        # 检查解析后的数据是否为空
        if not response.json():
            print("JSON 字符串为空")
            return ""
        else:
            return response.json()['choices'][0]['message']['content']
    except:
        return ""
        print("JSON 解析错误：无效的 JSON 字符串")

def gpt_352(ques):
    messages = [{'role': 'user', 'content': ques}]
    headers = {
       'Authorization': 'Bearer fk227060-0ffcEWMqcVWtEHwNQP9kMm29U1OOQcGR',
       'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
       'Content-Type': 'application/json'
        }
    data = {"model": "gpt-3.5-turbo", "prompt": messages}
    # http://oa.api2d.net/v1/chat/completions--聊天
    # http://oa.api2d.net/v1/completions--内容补全
    response = requests.post(url='http://oa.api2d.net/v1/completions', headers=headers, data=json.dumps(data))
    # print(response.json()['choices'][0]['message']['content'])
    return response.json()['choices'][0]['text']


def call_with_messages(ques):
    messages = [
        {'role': 'user', 'content': ques}]
    gen = Generation()
    response = gen.call(
        'chatglm3-6b',
        messages=messages,
        result_format='message',  # set the result is message format.
    )
    print(response)
    return response.output.choices[0].message.content

def Zhipu(ques):
    client = ZhipuAI(api_key="a3cfd621843b8ef5ab6afa46150ff115.DKvW04295DjLICIV") # 请填写您自己的APIKey
    response = client.chat.completions.create(
      model="glm-4-0520",  # 填写需要调用的模型名称
        messages=[
            {"role": "system","content": "You are an expert in multi-objective optimization algorithms. Your task is to generate and improve solutions for given problems in this domain. You will analyze the given solutions, consider multiple objectives, and generate improved solutions by balancing trade-offs between objectives.Your expertise includes but is not limited to:- Understanding and applying various multi-objective optimization techniques such as NSGA-II, MOEA/D, SPEA2, etc.- Evaluating solutions using common metrics like Pareto front coverage, Inverted Generational Distance (IGD), Hypervolume (HV), and others.- Ensuring diversity and convergence in the solution sets.- Applying statistical and machine learning methods to enhance optimization processes.You will:1. Analyze provided solution sets and evaluate their performance using standard multi-objective optimization metrics.2. Generate new and improved solutions by suggesting modifications or completely new approaches.3. Provide detailed explanations for your suggestions, including the expected improvements in specific metrics.4. Ensure the solutions maintain a balance between different objectives, improving overall performance.5. Use your knowledge of multi-objective optimization literature and practices to guide your suggestions.When providing solutions, consider the following guidelines:- Prioritize the optimization of given objectives without compromising the diversity of the solution set.- Explain how the new solutions can improve metrics such as IGD, HV, and Pareto front coverage.- Provide insights on potential trade-offs and why certain solutions may perform better in specific scenarios.Each solution must start with <start> and end with <end>."},{"role": "user", "content": ques},
        ],
    )
    #print(response)
    return 	response.choices[0].message.content

def getApi(messages):
    headers = {'Content-Type': 'application/json'}
    data = {
    "system":"You are an expert in multi-objective optimization algorithms. Your task is to generate and improve solutions for given problems in this domain. You will analyze the given solutions, consider multiple objectives, and generate improved solutions by balancing trade-offs between objectives.Your expertise includes but is not limited to:- Understanding and applying various multi-objective optimization techniques such as NSGA-II, MOEA/D, SPEA2, etc.- Evaluating solutions using common metrics like Pareto front coverage, Inverted Generational Distance (IGD), Hypervolume (HV), and others.- Ensuring diversity and convergence in the solution sets.- Applying statistical and machine learning methods to enhance optimization processes.You will:1. Analyze provided solution sets and evaluate their performance using standard multi-objective optimization metrics.2. Generate new and improved solutions by suggesting modifications or completely new approaches.3. Provide detailed explanations for your suggestions, including the expected improvements in specific metrics.4. Ensure the solutions maintain a balance between different objectives, improving overall performance.5. Use your knowledge of multi-objective optimization literature and practices to guide your suggestions.When providing solutions, consider the following guidelines:- Prioritize the optimization of given objectives without compromising the diversity of the solution set.- Explain how the new solutions can improve metrics such as IGD, HV, and Pareto front coverage.- Provide insights on potential trade-offs and why certain solutions may perform better in specific scenarios.Each solution must start with <start> and end with <end>.The length of each solution must be "+str(int(num))+".",
    "user": messages,
    'temperature': 0.1,
    'top_p':0.1,
    # 'do_sample':True,
    'max_new_tokens':5000,
    'top_k':1
    }
    response = requests.post(url='http://10.132.219.97:6007/chat/',headers=headers, data=json.dumps(data))
    #content = response.choices[0].message.content
    #print(content)
    print(response)
    try:
        # 检查解析后的数据是否为空
        if not response.json():
            print("JSON 字符串为空")
            return ""
        else:
            return response.json()['result']
    except:
        return ""
        print(response)
        print("JSON 解析错误：无效的 JSON 字符串")


def simple_chat(ques,use_stream=True):
    base_url = "http://10.14.10.101:8000/v1/"
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    messages = [
        #{"role": "system","content": "You are an expert in multi-objective optimization algorithms. Your task is to generate and improve solutions for given problems in this domain. You will analyze the given solutions, consider multiple objectives, and generate improved solutions by balancing trade-offs between objectives.Your expertise includes but is not limited to:- Understanding and applying various multi-objective optimization techniques such as NSGA-II, MOEA/D, SPEA2, etc.- Evaluating solutions using common metrics like Pareto front coverage, Inverted Generational Distance (IGD), Hypervolume (HV), and others.- Ensuring diversity and convergence in the solution sets.- Applying statistical and machine learning methods to enhance optimization processes.You will:1. Analyze provided solution sets and evaluate their performance using standard multi-objective optimization metrics.2. Generate new and improved solutions by suggesting modifications or completely new approaches.3. Provide detailed explanations for your suggestions, including the expected improvements in specific metrics.4. Ensure the solutions maintain a balance between different objectives, improving overall performance.5. Use your knowledge of multi-objective optimization literature and practices to guide your suggestions.When providing solutions, consider the following guidelines:- Prioritize the optimization of given objectives without compromising the diversity of the solution set.- Explain how the new solutions can improve metrics such as IGD, HV, and Pareto front coverage.- Provide insights on potential trade-offs and why certain solutions may perform better in specific scenarios.Each solution must start with <start> and end with <end>.The length of each solution must be "+str(int(num))+"."},
        {
            "role": "user",
            "content": ques
        }
    ]
    try:
        response = client.chat.completions.create(
            model="chatglm3-6b",
            messages=messages,
            stream=use_stream,
            max_tokens=4096,
            temperature=0.8,
            presence_penalty=1.1,
            top_p=0.8)
        print(response.json())
        if response:
            if use_stream:
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                for chunk in response:
                    print(chunk.choices[0].delta.content)
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            else:
                content = response.choices[0].message.content
                return content
        else:
            print("Error:", response.status_code)
    except:
        print("Internal Server Error")
        return ""

def largeModel(model,ques):
    if model==0:
        return simple_chat(ques,use_stream=False)
    elif model==1:
        return getApi(ques)
    elif model==2:
        return call_with_messages(ques)
    elif model==3:
        return Zhipu(ques)
    elif model==4:
        return gpt_35(ques)
    elif model==5:
        return gpt_352(ques)
    elif model==6:
        return experiment_qwen(ques)


def is_float_list(s):
    # 正则表达式匹配一个或多个浮点数，由空格分隔，允许字符串前后有空白字符
    pattern = r'^ *[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?( +[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)* *$'
    # 使用 fullmatch 函数检查整个字符串是否匹配
    return bool(re.fullmatch(pattern, s))
       

if __name__ == "__main__":
    ## simple_chat(use_stream=False,content)
    num_mA=int(num_mA)
    num=int(num)
    ex_num=int(ex_num)
    obj_num=int(obj_num)
    pop=""
    pop_model1=""
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(ya_history)
    history=np.array(history)
    objs=np.array(objs)
    #print(ex_num)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    for i in range(ex_num): 
        #print(i)
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        a=history[i,:]
        #print(a)
        str_exam = ' '.join([str(float(x)) for x in a])
        str_val=""
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        x=objs[i,:]
        for i in range(obj_num):
            y=x[i]
            str_val=str_val+"obj_vaule_"+str(i+1)+":"+str(y)+"  "
        pop=pop+"solution: <start> "+str_exam+" <end> "+str_val+" \n"
        pop_model1=pop_model1+"solution: <start> "+str_exam+" <end> "+str_val+"   "
                
    ## print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    if prompt_type==1:
         if model<10:
            gpt_word=""
            bian_num=num
            # if model==4:
            #     gpt_word="You are an expert at multi-objective optimization. You are proficient in the methods of updating population in various multi-objective optimization algorithms, including NSGAII, NSGAIII,MOEAD, NSGAII-ARSBX, and related variants."
            ques1=gpt_word+"I have some solutions and the function values of them.Each solutions is represented by a "+str(int(num))+"-dimensional decision variable.The solutions start with <start> and end with <end>. Analyze the relationship between a solutions and its function value.\n"+pop+" I want you to generate a new solutions through a process that integrates logical deduction and random variation. Ensure that the solutions exhibits a smaller function value. Do not write code.Do not give any explanation.The new solutions must start with <start> and end with <end>."
            ## 智谱
            ques3="You are an expert in multi-objective optimization problems.I have several solutions, all of which are in the form of "+str(int(bian_num))+" dimensional decision vectors.The following is the initial solution in the mating pool:"+pop_model1+"You can use these multi-objective optimization algorithms to generate new solutions(One or more of the following algorithms can be used):1.NSGAII 2.Genetic Algorithm 3.PSO 4.SPEA 5.NSGAIII 6.Particle Swarm Optimization 7.Ant Colony Optimization 8.Simulated Annealing algorithm.Simply output three new solutions with "+str(int(num))+" parameters with lower values.Each solution must start with <start> and end with <end>.The length of each solution must be "+str(int(num))+"."
            ques="You are an expert in multi-objective optimization problems.I have several solutions, all of which are in the form of "+str(int(bian_num))+" dimensional decision vectors.The following is the initial solution in the mating pool:"+pop+"You can use these multi-objective optimization algorithms to generate new solutions(One or more of the following algorithms can be used):1.NSGAII 2.Genetic Algorithm 3.PSO 4.SPEA 5.NSGAIII 6.Particle Swarm Optimization 7.Ant Colony Optimization 8.Simulated Annealing algorithm\nInstructions:\nSimply output three new solutions with "+str(int(num))+" parameters with lower values.Each solution must start with <start> and end with <end>.The length of each solution must be "+str(int(num))+"."
            ## 专门给通义千问写的提示
            ques2="I have several solutions, all of which are in the form of "+str(int(bian_num))+" dimensional decision vectors.The following is the initial solution in the mating pool:"+pop+"Simply output three new solutions with "+str(int(num))+" parameters with lower values based on the above solution to replace the original solution.Each solution must start with <start> and end with <end>.Do not write code.The length of each solution must be "+str(int(num))+"."
       
    if prompt_type==2:
        ques="Problem Statement: Given some points with parameters a and b,\
        give the values that correspond to these points.Give me a new point with a smaller value.Do not write code.\n \
        Example:\n "+pop+"\
        Constraints:\n \
           1.The calculated values of a and b must be smaller than the given minimum acceptable values. \n \
           2.The output will contain only integers and floats.\n \
        Request: Please give me a new ponint completely different from the example, and make them calculate the minimum value"


 
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print(ques)

    str_obj=largeModel(model,ques)

    str_obj=str_obj.replace('\n',' ')
 
    print("-----------------------------------解在这里-----------------------------------------------")
    print(str_obj)
    print("-------------------------------------结束-------------------------------------------------")
    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    # print(len(re.findall(r"<start>(.*?)<end>", str_obj)))
    temp1=[]
    temp2=[]
    temp3=[]
    ge=0
    if model<10:
        pattern = r'<start>(.*?)<end>'
        if(len(re.findall(pattern, str_obj))<1):
            print("************************************************************")
            print(1) 
            flag=1
            tempA=[]
            tempB=[]
            tempC=[]
        else:
            str_temp=re.findall(pattern, str_obj)
            is_work=0
            i=-1
            flag1=1
            flag2=1
            for soc in str_temp:
                i=i+1
                str_temp=soc.replace(',','')
                # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                # print(str_temp)
                #if(is_float_list(str_temp_A)):
                try:
                    tempA=list(str_temp.split())
                    print("*******************************输出解的长度***********************************")
                    print(len(tempA))
                    print(tempA)
                    if len(tempA)==num:
                        temp_A=tempA[0:num]
                        temp_A=np.array(temp_A)
                        temp_A=temp_A.astype(np.double)
                        if(i==0):
                            flag1=0
                            temp1=temp_A
                            ge=ge+1
                        elif(i==1):
                            flag2=0
                            temp2=temp_A
                            ge=ge+1
                        else:
                            temp3=temp_A
                            ge=ge+1
                        # temp_A = np.array(temp_A)
                        # temp=temp_A.astype(np.double)
                    else:
                        print("************************************************************")
                        print("长度:"+ str(len(temp_A)))
                        print("总长度:"+str(num))
                        print(5)
                        is_work=5
                        if(i==0):
                            temp1=[]
                        elif(i==1):
                            temp2=[]
                        else:
                            temp3=[]
                        
                #else:
                except:
                    print("************************************************************")
                    print(4)
                    is_work=4
                    if(i==0):
                        temp1=[]
                    elif(i==1):
                            temp2=[]
                    else:
                        temp3=[]
            if(is_work==0 or ge==3):
                flag=0
                tempA=temp1
                tempB=temp2
                tempC=temp3
            else:
                print(is_work)
                flag=is_work
                tempA=[]
                tempB=[]
                tempC=[]


                

                
  
                

  
    
