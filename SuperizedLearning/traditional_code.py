# In this program , i am writting tranditional python code to check spam emails...

def spam_email_cheack (email):
    spam_words = ["FREE", "WIN", "LOTTERY", "CLAIM", "PRIZE"]
    
    for word in spam_words:
        if word in email.upper():
            return "SPAM"
        
        return "NOT SPAM"
    

emails = [
    "WIN A FREE IPHONE TODAY",
    "TODAY IS THE TEAM METTING AT 3PM",
    "YOUR INVOICE IS ATTACHED",
]

print( "=== TRADITIONAL PROGRAMMING ===")
for email in emails:
    result = spam_email_cheack(email)
    print(f" '{email}' -> {result}" )       

    #  THE END ! 

