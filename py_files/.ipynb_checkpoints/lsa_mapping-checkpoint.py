import argparse
import pandas as pd 

parser = argparse.ArgumentParser()

if __name__ == "__main__":
    parser.add_argument('--lang', type=str, required=True, help='DE or EN')
    args = parser.parse_args()
    lang = args.lang.upper()
    folder = f"gen_files/{lang}/"

    # Load data
    df_tasks = pd.read_csv(f"{folder}LSA/tasks_topics.csv")[["taskId", "topic"]]
    df_tasks.columns = ["taskId", "taskTopic"]
    df_aspects = pd.read_csv(f"{folder}LSA/aspects_topics.csv")[["aspectId", "topic"]]
    df_aspects.columns = ["aspectId", "aspectTopic"]
    df_task_aspects = pd.read_csv(f"{folder}taskAspects.csv")

    df_task_aspects = pd.merge(df_task_aspects, df_tasks, on="taskId", how="inner")
    df_task_aspects = pd.merge(df_task_aspects, df_aspects, on="aspectId", how="inner")
    # columns of df_task_aspects = [taskId  aspectId  taskTopic  aspectTopic]

    # Mapping 1.0 
    df_map = df_task_aspects[["taskTopic", "aspectTopic"]]
    df_map1 = df_map.drop_duplicates().groupby("taskTopic")["aspectTopic"].apply(list).reset_index()
    df_map1["n"] = df_map1["aspectTopic"].apply(len)
    print("MAPPING 1")
    print(df_map1)
    
    # Mapping 2.0
    # For each task, assign the aspect_topics (from the task's aspects), and compute the probabilities of how often would topic i map to aspect_topic j 
    df_count = df_map.copy()
    df_count["count"] = 1
    df_count = df_count.groupby("taskTopic").count().reset_index()[["taskTopic", "count"]]

    df_count_per_topic = df_map.copy()
    df_count_per_topic["aspectCount"] = 1
    df_count_per_topic = df_count_per_topic.groupby(["taskTopic", "aspectTopic"]).count().reset_index()

    df_map2 = pd.merge(df_count_per_topic, df_count, on="taskTopic", how="left")
    df_map2["probability"] = df_map2["aspectCount"] / df_map2["count"]
    df_map2.rename(columns={"count":"totalCount"}, inplace=True)
    print("\nMAPPING 2")
    print(df_map2)

    print()
    
    # Define threshold t to decide which topics to keep
    # t = 1/n, n is the number of assigned aspectTopics
    df_map2 = pd.merge(df_map2, df_map1[["taskTopic", "n"]], on="taskTopic", how="left") 
    mask = df_map2["probability"] >= df_map2["n"].apply(lambda x: 1/x)
    df_map2 = df_map2[mask]
    df_map2.to_csv(f"{folder}/LSA/mapping.csv") 
    mapping = df_map2[["taskTopic", "aspectTopic"]].groupby("taskTopic")["aspectTopic"].apply(list).reset_index()
    
    # Mapping 3.0 
    # # Doing the same but considering the aspects instead     
    # df_count = df_map.copy()
    # df_count["count"] = 1
    # df_count = df_count.groupby("aspectTopic").count().reset_index()[["aspectTopic", "count"]]
    # df_count_per_topic = df_map.copy()
    # df_count_per_topic["taskCount"] = 1
    # df_count_per_topic = df_count_per_topic.groupby(["aspectTopic", "taskTopic"]).count().reset_index()
    # df_map3 = pd.merge(df_count_per_topic, df_count, on="aspectTopic", how="left")
    # df_map3["probability"] = df_map3["taskCount"] / df_map3["count"]
    
    # # # Define threshold t to decide which topics to keep
    # # # t = 1/n, n is the number of assigned aspectTopics
    # df_map_list = df_map.copy()
    # df_map_list = df_map_list.drop_duplicates().groupby("aspectTopic")["taskTopic"].apply(list).reset_index()
    # df_map_list["n"] = df_map_list["taskTopic"].apply(len)
    # df_map3 = pd.merge(df_map3, df_map_list[["aspectTopic", "n"]], on="aspectTopic", how="left") 
    
    # mask = df_map3["probability"] >= df_map3["n"].apply(lambda x: 1/x)
    # df_map3 = df_map3[mask]
    # # print(df_map3[["taskTopic", "aspectTopic"]].groupby("aspectTopic")["taskTopic"].apply(list).reset_index())
