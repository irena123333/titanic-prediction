import os
#import modal
    
BACKFILL=False
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_person(survived):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "pclass": [random.choice(range(1, 4))],
                       "sex": [random.choice([0, 1])],
                       "Age": [random.choice(range(1, 5))],
                       "fare": [random.choice(range(0, 4))],
                       "embarked": [random.choice(range(0, 3))],
                       "title": [random.choice(range(1, 6))],
                       "isalone": [random.choice([0,1])]
                      })
    df['survived'] = survived
    return df


def get_random_person():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    survived_df =  generate_person(1)
    unsurvived_df =  generate_person(0)
    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        person_df = survived_df
        print("Survived added")
    
    else:
        person_df = unsurvived_df
        print("Unsurvived added")

    return person_df



def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = pd.read_csv("/content/drive/MyDrive/titanic/serverless-ml-intro/cleaned_train.csv")
    else:
        titanic_df = get_random_person()

    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_survival_modal",
        version=1,
        primary_key=["Survived","PClass","Sex","Age","Fare","Embarked","Title","IsAlone"], 
        description="Titanic dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
