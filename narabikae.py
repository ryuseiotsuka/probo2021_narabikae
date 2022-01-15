#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt

class Q_learning(object):
    #初期化
    def __init__(self):
        # 並び替える数列
        #self.sequence = [3,2,4,1,5]
        self.action_space = 4
        self.done = False
        self.max_episode = 1000#エピソード数
        self.steps = 100#ステップ数
        self.alpha = 0.8#ステップサイズ
        #できうる数列は5の階乗通りで表すことができる
        self.q_table = np.random.uniform(low=0,high=1,size=(5**5,self.action_space))
        #描画用変数
        self.x = []
        self.y = []
        
    def moving_average(self, x, w):
        #描画用の移動平均導出
        return np.convolve(x, np.ones(w), 'valid') / w
    
    #エージェント
    def action(self,a):
        #数列の任意の桁とその隣を入れ替える
        temp = self.sequence[a+1]
        self.sequence[a+1] = self.sequence[a]
        self.sequence[a] = temp
    
    def decide_action(self,next_state,episode,q_table):
        #ϵ-greedy方策
        #ϵの確率でランダムに行動する
        f_prob = 0.75
        epsilon = f_prob * (1/(episode+1))
        if epsilon <= np.random.uniform(0,1):
            next_action = np.argmax(q_table[next_state])
        else:
            next_action = np.random.choice(range(4))#actionが4通りなので0から3の中からランダムに選択
        return next_action
    
    #状態        
    def get_state(self):
        #各数列の値を状態に変換
        #数列に入る数の最大値は5なので5進数で表現する
        state = (self.sequence[0]-1)*625+                (self.sequence[1]-1)*125+                (self.sequence[2]-1)*25+                (self.sequence[3]-1)*5+                (self.sequence[4]-1)*1
        return state
    
    #数列が[1,2,3,4,5]のときのみTrueにする
    def check(self):
        if self.sequence == [1,2,3,4,5]:
            return True
        else :
            return False
        
    def update_Qtable(self,state,action,reward,next_state):
        #Qテーブルの更新
        max_q_table = max(self.q_table[next_state])
        #print(state,action,reward,next_state)
        self.q_table[state,action] = (1-self.alpha) * self.q_table[state,action] + self.alpha * (reward + max_q_table) 
   
    #報酬
    def reward(self,done):   
        #並び替えが成功→1
        #そのほか→0
        if done == True:
            reward = 1
        else :
            reward = 0
        return reward
        
    def run(self):
        #実行
        #学習のループ
        for episode in range(self.max_episode):
            #エージェントの初期化
            print("-------------------")
            self.sequence = [3,2,4,1,5]
            state = self.get_state()
            action = np.argmax(self.q_table[state])
            for i in range(self.steps):
                print(self.sequence)
                self.action(action)#数列の入れ替え
                done = self.check()#並び替えが成功しているか確認
                reward = self.reward(done)
                next_state = self.get_state()
                self.update_Qtable(state,action,reward,next_state)#Qテーブルの更新
                action = self.decide_action(next_state,episode,self.q_table)#ϵ-greedy方策を使った行動決定
                state = next_state
                if done:
                    break
            #学習結果の表示
            print("episode: {0} steps: {1}".format(episode+1,i+1))
            self.x.append(episode)
            self.y.append(i)
        self.y = self.moving_average(self.y, 100)
        del self.x[:49]
        del self.x[-50:]
        plt.plot(self.x, self.y, label="steps")
        plt.legend()
        plt.show()
            
if __name__ == "__main__":
   Q_learning().run()     

