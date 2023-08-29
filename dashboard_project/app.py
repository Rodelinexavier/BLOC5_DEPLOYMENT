
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


### Config
st.set_page_config( page_title= "Get_Around",
                    page_icon="🚖",
                    layout="wide")

DATA_URL = ('https://getaround-deployment.s3.eu-west-3.amazonaws.com/get_around_delay_analysis.xlsx')

st.title(" :bar_chart: Dashboard")



def load_data():
    df = pd.read_excel(DATA_URL)
    return df

df=load_data()

#--------#
st.sidebar.header("Please filter here:")
checkin = st.sidebar.multiselect(
    "Select the checkin_type: ",
    options=df["checkin_type"].unique().tolist(),
    default=df["checkin_type"].unique().tolist()
)

state = st.sidebar.multiselect(
    "Select the state: ",
    options=df["state"].unique().tolist(),
    default=df["state"].unique().tolist()
)

df_selection = df.query(
    "checkin_type == @checkin & state== @state"
)


st.subheader("Load and showcase data")
data_load_data_state = st.text("Loading data...")
data_load_data_state.text("")

if st.checkbox("Show raw data"):
    st.subheader('Raw data')
    st.dataframe(df_selection)

#Preprocessing
return_checkout =[]
for x in df["delay_at_checkout_in_minutes"]:
    if x <= 0:
        return_checkout.append("on_time")
    else:
        return_checkout.append("late")

checkout = pd.DataFrame(return_checkout, columns=['check_out'])
df_selection['delay_checkout'] = checkout


state_canceled_df = df_selection[df_selection["state"]=="canceled"]
state_ended_df = df[df["state"] =="ended"]
late_arrival = df_selection[df_selection["delay_at_checkout_in_minutes"]>0]

def find_outliers_IQR(df):
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    IQR=q3-q1
    outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
    return outliers
# Source : How To Find Outliers in Data Using Python (and How To Handle Them):https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/

delay_checkout_outliers = find_outliers_IQR(late_arrival["delay_at_checkout_in_minutes"])

late_arrival.drop(delay_checkout_outliers.index, axis=0)

#Remove missing values
delay_cleaned = df_selection[df_selection["delay_at_checkout_in_minutes"].isna() == False]

#---MAIN PAGE ---#


#KPI's
total_cars = df_selection.shape[0]
#car_rental = int(len(df_selection['rental_id'].unique()))
average_delay = round(df_selection["delay_at_checkout_in_minutes"].mean())

left_column, middle_column, right_column= st.columns(3)
with left_column:
    st.subheader("Total Cars:")
    st.write(f" The companies contains {total_cars:,} cars")
#with middle_column:
    #st.subheader(" Rental:")
    #st.write(f"The number of car rental are {car_rental:,}")
with right_column:
    st.subheader(" Average delay:")
    st.write(f"The average delay are {average_delay:,} minutes")

st.markdown("______")
fig = px.histogram(delay_cleaned.sort_values("state"), x="state", y="delay_at_checkout_in_minutes", barmode="group")

st.plotly_chart(fig, use_container_width=True)

st.markdown("##")

st.markdown("_____")



st.subheader("Checkin_type")
fig = plt.figure(figsize=(20,10))
sns.countplot(x="checkin_type", data=delay_cleaned,palette="Set1")
st.pyplot(fig)


st.subheader("Checkin_type by state")
fig_2 = plt.figure(figsize=(20,10))
sns.countplot(data=df_selection, x="state",palette="Set1")
st.pyplot(fig_2)


st.markdown("___________")
#Mean by device
mean_by_device = round(delay_cleaned.groupby("checkin_type")["delay_at_checkout_in_minutes"].agg("mean"),2)
st.write(f"The mean of connection by connect is {mean_by_device[0]}")
print("=======")
st.write(f"The mean of connection by mobile is {mean_by_device[1]}")

st.markdown("________")
check_type = delay_cleaned.groupby(["checkin_type","delay_checkout"]).size().reset_index(name="quantity")
check_type["percent"] = [ x/check_type["quantity"].sum()* 100 for x in check_type["quantity"]]
check_type


