# SPAM EMAIL DETECTOR !
# Yeh program emails check kare ga.
# Spam hai ya nahi 

# step 01:

# skLearn ak libary hai ML ki 
# or us ka ander sa hum do cheze le raha hai 

# 1. countVectorizer - text ko number mai badle ga 
from sklearn.feature_extraction.text import CountVectorizer

# 2. MultinomialNB - yeh humara ML model hai.
# NB => naive Bayes - Ak simple and fast model.
from sklearn.naive_bayes import MultinomialNB

print ("Libaries import hop gai!..\n ")

#======================================================

# Step 02:

# ya wo emails hai jo hum model ko de ga or wo in sahi sekhke ga.

Traning_Emails = [
    "Win free iphone Now",
    "Claim Your Prize Today",
    "You won a lottery ticket",
    "Get Free Money and Balance",
    "Claim This offer NOW",
    "Get Free SMS AND MINTS Now",
    "Team meeting at 3pm Today",
    "Your invoice is attached",
    "Can you review my work",
    "Notes upload ho gaye hain",
]

# Yeh upar wali emails ke "sahi jawab" hain
# 1 = SPAM
# 0 = NOT SPAM
# sequence same hona chahiye upar wali list se

Labels = [ 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 ]

print (f"Total emails: {len(Traning_Emails)}")
print(f"Total Labels: {len(Labels)}\n")

#===================================================

# Step 03:

# Text ko numbers mai convert karna...
# countVectorizer ka tool banao
# abhi ya khali hai, is ko koi data nahi deya.
vectorizer = CountVectorizer()

# fit_transform = do kaam ak sath
# 1. fit => tamaamm emails mai sa unique words seehkho
# 2. transform => har email ko numbers ki row mai badlo 
X = vectorizer.fit_transform(Traning_Emails)

# dekho kitni rows and columns bani
print (f"Table ka size: {X.shape}")
# shape = (emails ki count , unique words ki count )

# dekho model na kon sa words sehkhe hai 
print (f"sehke howa words : {vectorizer.get_feature_names_out() }\n")

#=======================================================

# Step 04:

# Model BANAO AUR TRAIN KARO 
# phehle ak khlai model banaoo
# abhi ya kuch nahi janta ya ak khali student hai 
model = MultinomialNB()

# ab model ko train karna hoga 
# X wali number ka table dekar (emails ki) or labels bhi sath joka correct answers hai 
# model in dono dehk kar pattern sekhkha ga

model.fit(X, Labels)
print("Model train hogiya...\n")

# ====================================================

# Step 05:
# new emails ko predict karo

# ya wo emails hai jin ko model na kabhi nahi dehka , ab jum us sa poche ga ka ya spam hai ya nahi.

test_emails = [
    "Get free money now",       # spam lagti hai
    "Meeting tomorrow at 10am", # normal lagti hai
    "Claim your prize today",   # spam lagti hai
    "Please review my report",  # normal lagti hai
]

# Test email ko bhi numbers mai badalna ho ga 
# fit nai sarif transform karna ho ga 
# kiu ka vocab phele sa pata thi us ko 
X_test = vectorizer.transform(test_emails)

# ab model sa jawab mango 
results = model.predict(X_test)

# Results ko print karwaooo
print("\n PRINT THAT EMAIL ARE SPAM OR NOT: \n")
for email , result in zip(test_emails, results):
    if result == 1:
        label = "SPAM"
    else:
        label = "Not SPAM"
    print (f" {email} -> {label} \n")

# ================== THE END =============== #   