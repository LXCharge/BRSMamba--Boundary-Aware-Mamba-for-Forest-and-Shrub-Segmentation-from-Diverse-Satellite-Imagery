import datetime
import timm

if __name__ == "__main__":
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    print(current_date)
    model_list = timm.list_models(pretrained=True)
    print(model_list)
