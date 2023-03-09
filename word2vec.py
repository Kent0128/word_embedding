import math
import numpy as np
import tensorflow as tf
with open("xjp.txt") as f:
    text=str(f.read()).replace(',','').split(".")
window_size=25
embedding_size=7#vector 長度
num_sampled = 3
context_pair=[]
word_set=set()

def __is_bounded(direction,range,index,tokens_leng):
    cover = range*direction
    if cover+index<0 or cover+index >= tokens_leng:
        return True
    else:
        return False

def get_context(tokens, window_size):
    context_pair = []
    for i, token in enumerate(tokens):
        for j in range(1, window_size+1):
            if not __is_bounded(1,j,i,len(tokens)):
                context_pair.append((tokens[i],tokens[i+j]))
            if not __is_bounded(-1,j,i,len(tokens)):
                context_pair.append((tokens[i],tokens[i-j]))
    return context_pair

def __get_word_set(tokens):
    word_set = set()
    for token in tokens:
        word_set.add(token)
    return word_set

def __get_word_index(word_set):
    word_index_dic = dict()
    for i,word in enumerate(word_set):
        word_index_dic[word] = i
    return word_index_dic

def generate_batch(context_pair,batch_size):
    batch_list =[]
    batch=[]
    for i,pair in enumerate(context_pair):

        if i %batch_size==0 and i !=0:
            batch_list.append(batch)
            batch = []

        batch.append(pair)
    return batch_list

def get_vec(word,session):
    return session.run(embeddings[word_index_dic[word]])

def get_cos_similarity(vec1, vec2):
    vec1_leng=0
    for value in vec1:
        vec1_leng+=(value*value)
    vec1_leng=math.sqrt(vec1_leng)
    vec2_leng=0
    for value in vec2:
        vec2_leng+=(value*value)
    vec2_leng=math.sqrt(vec2_leng)
    product=np.dot(vec1,vec2)

    return product/(vec1_leng*vec2_leng)

def find_cloest_word(word_set,session,target_word):
    sim = 0.0
    vec1 = get_vec(target_word,session)
    result = ''
    for word in word_set:
        if word == target_word:
            continue
        vec2 = get_vec(word,session)
        tmp_sim=__sim(vec1, vec2)
        print('%s : %s : %s' %(target_word,word,tmp_sim))
        if tmp_sim>sim:
            sim = tmp_sim
            result = word
    return result

def __sim(vec1, vec2):
    return (1 - math.acos(get_cos_similarity(vec1,vec2)) / math.pi)

for sentence in text:
    tokens=sentence.lower().split(" ")
    context_pair += get_context(tokens,window_size)
    tmp_word_set=__get_word_set(tokens)
    for word in tmp_word_set:
        word_set.add(word)
print(context_pair)
word_index_dic=__get_word_index(word_set)
word_size = len(word_set)
batch_size = len(context_pair)
inputs = [word_index_dic[x[0]] for x in context_pair]
labels = [[word_index_dic[x[1]]] for x in context_pair]
tf.compat.v1.disable_eager_execution()
train_inputs = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size,1])
embeddings = tf.Variable(tf.compat.v1.random_uniform([word_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
nce_weights = tf.Variable(tf.compat.v1.truncated_normal([word_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([word_size]))
loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=word_size))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
session = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
session.run(init)
#print(word_index_dic)
for iteration in range(0,10000):
    total_loss = 0
    feed_dict = {train_inputs: inputs, train_labels: labels}
    _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)
    print('%s: loss: %s' %(iteration,cur_loss))
print(find_cloest_word(word_set, session,"funk"))
#print(find_cloest_word(word_set, session, 'he'))
#print(find_cloest_word(word_set, session, 'He'))

#data=Rock a genre of popular music that originated in the 1950s, characterized by heavy use of electric guitars, bass guitar, and drums, as well as often incorporating elements of blues, jazz, and folk music.Pop a genre of popular music that originated in the 1950s, characterized by a catchy melody, simple chord structure, and a strong beat.
#Hip-hop a genre of popular music that originated in the 1970s in the United States, characterized by the use of rap and DJing, as well as the use of samples from other songs.Electronic a genre of popular music that originated in the 1980s and is characterized by the use of electronic instruments and electronic music technology, such as synthesizers, drum machines, and digital audio workstations.Jazz a genre of popular music that originated in the African-American communities of New Orleans in the late 19th and early 20th centuries, characterized by a heavy emphasis on improvisation and the use of a wide range of musical instruments.Classical a genre of music that originated in Europe in the 19th century and is characterized by its use of orchestral instruments and a formal musical structure.lues a genre of popular music that originated in the African-American communities of the southern United States in the late 19th and early 20th centuries, characterized by its use of the blues scale and its lyrics, which often deal with themes of pain and hardship.Country a genre of popular music that originated in the southern United States in the 1920s, characterized by its use of stringed instruments, a strong emphasis on lyrics, and a twangy, nasal vocal style.Folk a genre of popular music that originated in the traditional music of various cultures, characterized by its use of acoustic instruments and its emphasis on storytelling and social commentary.R&B a genre of popular music that originated in the African-American communities of the United States in the 1940s, characterized by its use of soulful vocals and a strong emphasis on rhythm and blues.
#Soul a genre of music that originated in the African-American communities of the United States in the 1950s and 1960s, characterized by a strong emphasis on vocals and the use of gospel and R&B influences.Funk a genre of music that originated in the African-American communities of the United States in the mid-1960s and is characterized by a heavy, syncopated beat, electric bass and guitar, and the use of horns and keyboards.

