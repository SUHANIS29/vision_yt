import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from googleapiclient.discovery import build
from utils import fetch_channel_data, get_channel_id_from_url
import plotly.express as px
from utils import fetch_multi_year_video_data
import numpy as np
from datetime import datetime, timedelta
import calendar
from textblob import TextBlob
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="YouTube Channel Analyzer", layout="wide")
st.title("✨ VisionYT ✨")
st.title("YouTube Channel Analyzer &amp; Comparator - 2025 Growth Strategy")

# Display Analysis Goals
with st.expander("📊 Analysis Goals &amp; Insights"):
    st.markdown("""
**This tool analyzes:**
- 📈 **Trending Niches in 2025** - Which content categories perform best
- ⏱️ **Optimal Video Length** - Short-form vs Long-form performance
- 📅 **Upload Strategy** - Best times/days and frequency impact
- 🎯 **Content Optimization** - Title keywords and thumbnail correlation
- 🎭 **Audience Sentiment** - Comment analysis for loyalty insights
- 🚀 **Growth Strategies** - Data-driven recommendations
- 📊 **Individual Channel Analysis** - Personalized recommendations for each channel
- 👥 **Subscriber Comparison** - Channel growth comparison
""")

# api_key = "AIzaSyD0xaOHtlaE_hM1I5CIKXtUASYLSoSCYtI"
# api_key="AIzaSyBAzHVdwfjdkSOwV3gkQt5JHURIqOraO4Q"
api_key ="AIzaSyAMfCWD_G8x1X9Gs4FnDVhKoaLFAlPn1pQ"
# Helper function to get subscriber count
def get_subscriber_count(channel_id, api_key):
    """Get subscriber count for a channel"""
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        response = youtube.channels().list(
            part="statistics",
            id=channel_id
        ).execute()
        if response['items']:
            stats = response['items'][0]['statistics']
            return int(stats.get('subscriberCount', 0))
        return 0
    except Exception:
        return 0

# Input
st.markdown("Paste one or more YouTube channel URLs below (one per line):")
urls = st.text_area("Channel URLs")
url_list = [url.strip() for url in urls.split("\n") if url.strip()]

if st.button("Analyze Channels"):
    if not url_list:
        st.warning("Please enter at least one URL.")
    else:
        all_data = []
        subscriber_data = []
        youtube = build("youtube", "v3", developerKey=api_key)

        for url in url_list:
            try:
                ch_id = get_channel_id_from_url(url, youtube)
                df = fetch_channel_data(ch_id, api_key, max_videos=100)
                sub_count = get_subscriber_count(ch_id, api_key)
                subscriber_data.append({
                    'Channel': df['Channel'].iloc[0],
                    'Subscribers': sub_count,
                    'Channel_ID': ch_id
                })
                all_data.append(df)
                st.success(f"Fetched data for **{df['Channel'].iloc[0]}** - {sub_count:,} subscribers")
            except Exception as e:
                st.error(f"Error fetching for {url}: {e}")

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df["Duration_sec"] = pd.to_timedelta(final_df["Duration"]).dt.total_seconds()
            final_df["Publish Time"] = pd.to_datetime(final_df["Publish Time"])
            final_df["Day"] = final_df["Publish Time"].dt.day_name()
            final_df["Month"] = final_df["Publish Time"].dt.to_period("M").astype(str)
            final_df["Year"] = final_df["Publish Time"].dt.year
            final_df["Hour"] = final_df["Publish Time"].dt.hour
            final_df["isShort"] = final_df["Duration_sec"] < 60

            # Enhanced keyword analysis for 2025 trends
            trending_keywords_2025 = [
                "ai", "artificial intelligence", "chatgpt", "automation", "tech", "crypto", "blockchain",
                "sustainability", "green", "climate", "mental health", "wellness", "productivity",
                "remote work", "side hustle", "passive income", "investing", "stocks", "real estate",
                "fitness", "workout", "diet", "recipe", "gaming", "esports", "streaming",
                "top 10", "shocking", "must watch", "revealed", "secret", "facts", "you won't believe", "crazy"
            ]
            def has_trending_keyword(title):
                title_lower = title.lower()
                return any(keyword in title_lower for keyword in trending_keywords_2025)
            final_df["Has Trending Keywords"] = final_df["Title"].apply(has_trending_keyword)
            
            # Clickbait analysis
            keywords = ["top 10", "shocking", "must watch", "revealed", "secret", "facts", "you won't believe", "crazy"]
            def has_keyword(title):
                title_lower = title.lower()
                return any(keyword in title_lower for keyword in keywords)
            final_df["Clickbait Title"] = final_df["Title"].apply(has_keyword)

            # Engagement Rate Calculation
            #  Engagement Rate (%)=( Likes+Comments)÷Views×100

            if "Engagement Rate (%)" not in final_df.columns:
                final_df["Engagement Rate (%)"] = ((final_df["Likes"] + final_df["Comments"]) / final_df["Views"]) * 100
                final_df["Engagement Rate (%)"].replace([float("inf"), -float("inf")], 0, inplace=True)
                final_df["Engagement Rate (%)"].fillna(0, inplace=True)

            # SUBSCRIBER COMPARISON SECTION
            if len(subscriber_data) > 1:
                st.markdown("## 👥 Subscriber Count Comparison")
                subscriber_df = pd.DataFrame(subscriber_data)
                col1, col2 = st.columns(2)
                with col1:
                    highest_subs = subscriber_df.loc[subscriber_df['Subscribers'].idxmax()]
                    st.metric("🏆 Highest Subscribers", f"{highest_subs['Subscribers']:,}", highest_subs['Channel'])
                with col2:
                    sorted_df = subscriber_df.sort_values(by='Subscribers', ascending=False)
                    if len(sorted_df) > 1:
                        second_subs = sorted_df.iloc[1]

            st.subheader("📋Video Data Overview")
            st.dataframe(final_df)

            # ENHANCED NLP CONTENT CATEGORIZATION
            st.markdown("## 🎯2025 Content Category Analysis")
            train_data = {
                "Title": [
                    "Top 10 Amazing Facts About Space", "How AI Will Change Everything", "My Morning Routine for Success",
                    "Unboxing the Latest iPhone 15", "Shocking Secrets of Ancient Egypt", "Funny Prank on My Sister",
                    "Machine Learning Tutorial for Beginners", "Daily Vlog: Trip to Japan", "Cryptocurrency Explained Simply",
                    "Best Workout for Weight Loss", "Healthy Breakfast Recipes", "Gaming Setup Tour 2025",
                    "Stock Market Predictions", "Mental Health Tips", "Sustainable Living Guide",
                    "Remote Work Productivity Hacks", "Side Hustle Ideas 2025", "Climate Change Solutions",
                    "Motivational Speech by CEO", "Technology Trends 2025", "Fitness Transformation Story",
                    "Recipe: Quick 15-Minute Meal", "Gaming Review: New RPG", "Investment Strategy Explained",
                    "Meditation for Beginners", "DIY Home Improvement", "Fashion Haul Winter 2025",
                    "Movie Review: Latest Blockbuster", "Travel Guide to Tokyo", "Comedy Sketch Compilation",
                    "Educational Documentary", "Entertainment Tonight", "Product Review iPhone", "My Daily Vlog"
                ],
                "Category": [
                    "Educational", "Tech/AI", "Lifestyle", "Tech Review", "Educational", "Entertainment",
                    "Tech/AI", "Vlog", "Finance", "Fitness", "Food", "Gaming",
                    "Finance", "Health/Wellness", "Lifestyle", "Productivity", "Finance", "Educational",
                    "Motivation", "Tech/AI", "Fitness", "Food", "Gaming", "Finance",
                    "Health/Wellness", "DIY", "Fashion", "Review", "Travel", "Entertainment",
                    "Educational", "Entertainment", "Review", "Vlog"
                ]
            }

            df_train = pd.DataFrame(train_data)
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            X_train = vectorizer.fit_transform(df_train["Title"])
            y_train = df_train["Category"]
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(X_train, y_train)

            def nlp_title_category(title):
                x_test = vectorizer.transform([title])
                return classifier.predict(x_test)[0]
            final_df["Content Category"] = final_df["Title"].apply(nlp_title_category)
            
            # RESEARCH QUESTION 1: Which niches are trending in 2025?
            category_perf = final_df.groupby("Content Category").agg({
                "Views": ["mean", "median", "count", "sum"],
                "Likes": "mean",
                "Comments": "mean",
                "Duration_sec": "mean"
            }).round(0)
            category_perf.columns = ["Avg Views", "Median Views", "Video Count", "Total Views", "Avg Likes", "Avg Comments", "Avg Duration"]
            category_perf["Engagement Rate"] = ((category_perf["Avg Likes"] + category_perf["Avg Comments"]) / category_perf["Avg Views"] * 100).round(2)
            category_perf["Growth Potential"] = (category_perf["Avg Views"] * category_perf["Engagement Rate"] / 100).round(0)
            category_perf = category_perf.sort_values("Growth Potential", ascending=False)

            st.markdown("### 🏆 Top Trending Content Categories in 2025")
            st.dataframe(category_perf)
            fig_cat = px.bar(category_perf.reset_index(), x="Content Category", y="Growth Potential",
                             color="Engagement Rate", title="🚀 2025 Trending Niches by Growth Potential",
                             text_auto=".2s", hover_data=["Video Count", "Avg Views"])
            fig_cat.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_cat, use_container_width=True)

            top_niche = category_perf.index[0]
            st.success(f"**Answer:** {top_niche} is the top trending niche in 2025 with {category_perf.loc[top_niche, 'Growth Potential']:,.0f} growth potential score and {category_perf.loc[top_niche, 'Engagement Rate']:.1f}% engagement rate.")

            # UPLOAD FREQUENCY ANALYSIS
            st.markdown("## 📅 Upload Strategy Analysis")
            # Calculate upload frequency per channel
            upload_freq = final_df.groupby("Channel").agg({
                "Publish Time": ["min", "max", "count"]
            })
            upload_freq.columns = ["First Upload", "Last Upload", "Total Videos"]
            upload_freq["Days Active"] = (upload_freq["Last Upload"] - upload_freq["First Upload"]).dt.days + 1
            upload_freq["Videos Per Day"] = (upload_freq["Total Videos"] / upload_freq["Days Active"]).round(3)
            upload_freq["Upload Frequency"] = upload_freq["Videos Per Day"].apply(
                lambda x: "Daily+" if x >= 1 else "Weekly" if x >= 0.14 else "Monthly" if x >= 0.03 else "Irregular"
            )

            # ENHANCED INDIVIDUAL CHANNEL UPLOAD PATTERN ANALYSIS
            for channel_name in final_df["Channel"].unique():
                channel_df = final_df[final_df["Channel"] == channel_name].copy()
                st.markdown(f"#### 📈 Upload Pattern Analysis - {channel_name}")

                # 1. Monthly Upload Consistency
                monthly_uploads = channel_df.groupby(channel_df["Publish Time"].dt.to_period("M")).agg({
                    "Views": ["mean", "count"],
                    "Engagement Rate (%)": "mean"
                }).round(2)
                monthly_uploads.columns = ["Avg Views", "Video Count", "Avg Engagement"]
                monthly_uploads.index = monthly_uploads.index.astype(str)

                if len(monthly_uploads) > 1:
                    fig_monthly = make_subplots(
                        rows=1, cols=1,
                        specs=[[{"secondary_y": True}]]
                    )
                    fig_monthly.add_trace(
                        go.Bar(x=monthly_uploads.index, y=monthly_uploads["Video Count"],
                            name="Videos", marker_color='rgba(135, 206, 235, 0.6)'),
                        row=1, col=1
                    )
                    fig_monthly.add_trace(
                        go.Scatter(x=monthly_uploads.index, y=monthly_uploads["Avg Views"],
                                  mode='lines+markers', name="Avg Views",
                                  line=dict(color='red', width=3), marker=dict(size=8)),
                        row=1, col=1, secondary_y=True
                    )
                    fig_monthly.update_xaxes(title_text="Month", row=1, col=1)
                    fig_monthly.update_yaxes(title_text="Video Count", row=1, col=1)
                    fig_monthly.update_yaxes(title_text="Average Views", secondary_y=True, row=1, col=1)
                    fig_monthly.update_layout(height=400, showlegend=True, title=f"Monthly Upload Consistency - {channel_name}")
                    st.plotly_chart(fig_monthly, use_container_width=True)

                # 2. Day-wise Upload Strategy - SIMPLIFIED VISUALIZATION
                channel_df['WeekDay'] = channel_df['Publish Time'].dt.day_name()
                channel_df['Week'] = channel_df['Publish Time'].dt.isocalendar().week

                # Create aggregated data for day and hour performance
                day_hour_performance = channel_df.groupby(['WeekDay', 'Hour'])['Views'].mean().reset_index()

                if not day_hour_performance.empty:
                    # Create grouped bar chart instead of heatmap
                    fig_simple = px.bar(
                        day_hour_performance, 
                        x='WeekDay', 
                        y='Views',
                        color='Hour',
                        barmode='group',
                        title=f"Best Upload Times by Day and Hour - {channel_name}",
                        labels=dict(WeekDay="Day of Week", Views="Average Views", Hour="Upload Hour")
                    )
                    
                    # Order days properly
                    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    fig_simple.update_xaxes(categoryorder='array', categoryarray=weekday_order)
                    
                    st.plotly_chart(fig_simple, use_container_width=True)

                # 3. Upload Streak Analysis for EACH CHANNEL
                st.markdown("### 📅 Upload Consistency Analysis")
                channel_df_sorted = channel_df.sort_values('Publish Time')
                channel_df_sorted['Days_Between'] = channel_df_sorted['Publish Time'].diff().dt.days
                avg_gap = channel_df_sorted['Days_Between'].mean()
                consistency_score = 100 / (1 + avg_gap) if not pd.isna(avg_gap) else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📅 Avg Days Between Uploads", f"{avg_gap:.1f}" if not pd.isna(avg_gap) else "N/A")
                with col2:
                    st.metric("🎯 Consistency Score", f"{consistency_score:.1f}%")
                with col3:
                    freq_category = upload_freq.loc[channel_name, "Upload Frequency"] if channel_name in upload_freq.index else "N/A"
                    st.metric("📊 Upload Pattern", freq_category)
                
                # Additional consistency insights
                if not pd.isna(avg_gap):
                    if avg_gap <= 1:
                        consistency_level = "🟢 Excellent (Daily)"
                    elif avg_gap <= 3:
                        consistency_level = "🟡 Good (Every 2-3 days)"
                    elif avg_gap <= 7:
                        consistency_level = "🟠 Moderate (Weekly)"
                    else:
                        consistency_level = "🔴 Irregular"
                    
                    st.info(f"**Consistency Level:** {consistency_level}")
                
                st.markdown("---")  # Separator between channels

            # Uploads by Day
            if len(final_df["Channel"].unique()) > 1:
                day_uploads = final_df.groupby(["Channel", "Day"]).size().unstack().fillna(0)
                st.markdown("### 📊 Uploads by Day of Week - All Channels")
                fig_days = px.bar(day_uploads.T, title="Upload Distribution by Day of Week")
                fig_days.update_layout(xaxis_title="Day of Week", yaxis_title="Number of Uploads")
                st.plotly_chart(fig_days, use_container_width=True)

            # VIDEO LENGTH ANALYSIS
            st.markdown("## ⏱️ Video Length Strategy Analysis")
            
            # Define video length categories
            def categorize_length(duration_sec):
                if duration_sec < 60:
                    return "Shorts (<1min)"
                elif duration_sec < 300:
                    return "Short (1-5min)"
                elif duration_sec < 600:
                    return "Medium (5-10min)"
                elif duration_sec < 1200:
                    return "Long (10-20min)"
                else:
                    return "Very Long (20min+)"
            
            final_df["Length Category"] = final_df["Duration_sec"].apply(categorize_length)
            
            length_analysis = final_df.groupby("Length Category").agg({
                "Views": ["mean", "median"],
                "Likes": "mean",
                "Comments": "mean",
                "Title": "count"
            }).round(0)
            length_analysis.columns = ["Avg Views", "Median Views", "Avg Likes", "Avg Comments", "Video Count"]
            length_analysis["Engagement Rate"] = ((length_analysis["Avg Likes"] + length_analysis["Avg Comments"]) / length_analysis["Avg Views"] * 100).round(2)
            
            # Order by video length
            length_order = ["Shorts (<1min)", "Short (1-5min)", "Medium (5-10min)", "Long (10-20min)", "Very Long (20min+)"]
            length_analysis = length_analysis.reindex(length_order).dropna()
            
            st.markdown("### 📏 Video Length Performance Analysis")
            st.dataframe(length_analysis)
            
            fig_length = px.bar(length_analysis.reset_index(), x="Length Category", y="Avg Views",
                              color="Engagement Rate", title="📊 Performance by Video Length",
                              text_auto=".2s")
            st.plotly_chart(fig_length, use_container_width=True)

            # ==========================================
            # 🎯 INDIVIDUAL CHANNEL ANALYSIS SECTION
            # ==========================================
            st.markdown("# 🎯 Individual Channel Analysis &amp; Recommendations")
            st.markdown("---")

            # NEW: ENHANCED CHANNEL COMPARISON VISUALIZATIONS
            if len(final_df["Channel"].unique()) > 1:
                st.markdown("## 📊 Enhanced Channel Performance Comparison")
                
                # Create comparison data
                channel_comparison = final_df.groupby("Channel").agg({
                    "Views": ["mean", "sum"],
                    "Engagement Rate (%)": "mean",
                    "Likes": "mean",
                    "Comments": "mean",
                    "Title": "count"
                }).round(2)
                
                channel_comparison.columns = ["Avg Views", "Total Views", "Avg Engagement Rate", "Avg Likes", "Avg Comments", "Video Count"]
                channel_comparison = channel_comparison.reset_index()
                
                # Combined Performance Dashboard
                fig_combined = make_subplots(
                        rows=2, cols=2,
                        
                        subplot_titles=('Engagement Rate (%)', 'Average Views', 'Total Views', 'Video Count'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                
                # Add traces
                fig_combined.add_trace(
                        go.Bar(x=channel_comparison["Channel"], y=channel_comparison["Avg Engagement Rate"], 
                               name="Engagement Rate", marker_color='lightblue'),
                        row=1, col=1
                    )
                    
                fig_combined.add_trace(
                        go.Bar(x=channel_comparison["Channel"], y=channel_comparison["Avg Views"], 
                               name="Avg Views", marker_color='lightgreen'),
                        row=1, col=2
                    )
                
                fig_combined.add_trace(
                        go.Bar(x=channel_comparison["Channel"], y=channel_comparison["Total Views"], 
                               name="Total Views", marker_color='lightcoral'),
                        row=2, col=1
                    )
                
                # This seems to be missing in the original code, so adding it for completeness
                fig_combined.add_trace(
                        go.Bar(x=channel_comparison["Channel"], y=channel_comparison["Video Count"], 
                               name="Video Count", marker_color='lightsalmon'),
                        row=2, col=2
                    )

                fig_combined.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_combined, use_container_width=True)
                
            # Helper functions for individual analysis
            def get_best_upload_time(channel_df):
                """Find best upload hour and day for a channel"""
                hour_performance = channel_df.groupby("Hour")["Views"].mean().sort_values(ascending=False)
                day_performance = channel_df.groupby("Day")["Views"].mean().sort_values(ascending=False)
                
                best_hour = hour_performance.index[0] if len(hour_performance) > 0 else "Not enough data"
                best_day = day_performance.index[0] if len(day_performance) > 0 else "Not enough data"
                
                return best_hour, best_day, hour_performance, day_performance

            def analyze_shorts_vs_long(channel_df):
                """Analyze shorts vs long-form performance"""
                if channel_df["isShort"].sum() == 0:
                    return "No Shorts", "Focus on Long-form", {}
                
                shorts_vs_long = channel_df.groupby("isShort").agg({
                    "Views": "mean",
                    "Likes": "mean",
                    "Comments": "mean",
                    "Title": "count"
                }).round(0)
                
                shorts_vs_long.index = ["Long-form", "Shorts"]
                shorts_vs_long["Engagement Rate"] = ((shorts_vs_long["Likes"] + shorts_vs_long["Comments"]) / shorts_vs_long["Views"] * 100).round(2)
                
                if "Shorts" in shorts_vs_long.index and "Long-form" in shorts_vs_long.index:
                    if shorts_vs_long.loc["Shorts", "Views"] > shorts_vs_long.loc["Long-form", "Views"]:
                        recommendation = "Focus more on Shorts - they get higher views"
                    else:
                        recommendation = "Long-form content performs better"
                else:
                    recommendation = "Not enough data for comparison"
                
                return shorts_vs_long, recommendation, shorts_vs_long.to_dict()

            def get_fastest_growing_category(channel_df):
                """Find which content category grows fastest for this channel"""
                if len(channel_df) < 10:
                    return "Not enough data", {}
                
                category_growth = channel_df.groupby("Content Category").agg({
                    "Views": ["mean", "count"],
                    "Engagement Rate (%)": "mean"
                }).round(0)
                
                category_growth.columns = ["Avg Views", "Video Count", "Avg Engagement"]
                category_growth = category_growth[category_growth["Video Count"] >= 2]  # At least 2 videos
                
                if len(category_growth) > 0:
                    fastest_growing = category_growth.sort_values("Avg Views", ascending=False).index[0]
                    return fastest_growing, category_growth.to_dict()
                else:
                    return "Not enough data", {}

            # Individual channel analysis loop
            for channel_name in final_df["Channel"].unique():
                channel_df = final_df[final_df["Channel"] == channel_name].copy()
                
                st.markdown(f"## 📊 Analysis for: **{channel_name}**")
                
                # Basic metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Videos", len(channel_df))
                with col2:
                    st.metric("Avg Views", f"{int(channel_df['Views'].mean()):,}")
                with col3:
                    st.metric("Avg Engagement Rate", f"{channel_df['Engagement Rate (%)'].mean():.2f}%")
                with col4:
                    total_views = channel_df["Views"].sum()
                    st.metric("Total Views", f"{total_views:,}")

                # 1. BEST UPLOAD TIME ANALYSIS
                st.markdown("### ⏰ Best Upload Time Analysis")
                best_hour, best_day, hour_perf, day_perf = get_best_upload_time(channel_df)
                
                time_col1, time_col2 = st.columns(2)
                with time_col1:
                    st.markdown(f"**Best Upload Hour:** {best_hour}:00")
                    if len(hour_perf) > 1:
                        fig_hour = px.bar(x=hour_perf.index, y=hour_perf.values, 
                                        title=f"Views by Upload Hour - {channel_name}")
                        fig_hour.update_xaxes(title="Hour of Day", type='category')
                        fig_hour.update_yaxes(title="Average Views")
                        st.plotly_chart(fig_hour, use_container_width=True)
                
                with time_col2:
                    st.markdown(f"**Best Upload Day:** {best_day}")
                    if len(day_perf) > 1:
                        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                        day_perf = day_perf.reindex(day_order).dropna()
                        day_perf_df = pd.DataFrame({
                            'Day': day_perf.index,
                            'Avg_Views': day_perf.values
                        })
                        
                        fig_day = px.bar(day_perf_df, x="Day", y="Avg_Views",
                                       title=f"Views by Upload Day - {channel_name}")
                        fig_day.update_xaxes(title="Day of Week")
                        fig_day.update_yaxes(title="Average Views")
                        st.plotly_chart(fig_day, use_container_width=True)
                    else:
                        st.info("Not enough data for day-wise analysis")

                # 2. SHORTS VS LONG-FORM ANALYSIS
                st.markdown("### 📱 Shorts vs Long-form Analysis")
                shorts_analysis, shorts_rec, shorts_data = analyze_shorts_vs_long(channel_df)
                
                if isinstance(shorts_analysis, pd.DataFrame):
                    st.dataframe(shorts_analysis)
                    
                    fig_shorts = px.bar(shorts_analysis.reset_index(), x="index", y="Views",
                                      color="Engagement Rate", title=f"Shorts vs Long-form - {channel_name}")
                    st.plotly_chart(fig_shorts, use_container_width=True)
                
                st.success(f"**Recommendation:** {shorts_rec}")

                # 3. FASTEST GROWING CONTENT CATEGORY
                st.markdown("### 🚀 Which Content Type Grows Fastest?")
                fastest_category, category_data = get_fastest_growing_category(channel_df)
                
                if fastest_category != "Not enough data":
                    channel_categories = channel_df.groupby("Content Category").agg({
                        "Views": "mean",
                        "Engagement Rate (%)": "mean",
                        "Title": "count"
                    }).round(2)
                    channel_categories.columns = ["Avg Views", "Avg Engagement", "Video Count"]
                    channel_categories = channel_categories[channel_categories["Video Count"] >= 2]
                    
                    if len(channel_categories) > 0:
                        st.dataframe(channel_categories.sort_values("Avg Views", ascending=False))
                        
                        fig_cat = px.bar(channel_categories.reset_index(), 
                                       x="Content Category", y="Avg Views",
                                       color="Avg Engagement",
                                       title=f"Content Category Performance - {channel_name}")
                        fig_cat.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_cat, use_container_width=True)
                    
                    st.success(f"**Fastest Growing Category:** {fastest_category}")
                else:
                    st.warning("Not enough data to determine fastest growing category")

                # 5. TOP PERFORMING VIDEOS FOR THIS CHANNEL
                st.markdown("### 🏆 Top 5 Performing Videos")
                top_videos = channel_df.sort_values("Views", ascending=False).head(5)
                top_videos_display = top_videos[["Title", "Views", "Likes", "Comments", "Content Category", "Length Category"]].copy()
                st.dataframe(top_videos_display.style.format({"Views": "{:,}", "Likes": "{:,}", "Comments": "{:,}"}))

                st.markdown("---")  # Separator between channels

            # ==========================================
            # 🔄 MULTI-CHANNEL COMPARISON & EXECUTIVE SUMMARY
            # ==========================================
            if len(final_df["Channel"].unique()) > 1:
                st.markdown("# 🔄 Multi-Channel Comparison &amp; Competition Analysis")
                
                # Create comparison metrics
                comparison_data = []
                
                for channel_name in final_df["Channel"].unique():
                    channel_df = final_df[final_df["Channel"] == channel_name].copy()
                    
                    best_hour, best_day, _, _ = get_best_upload_time(channel_df)
                    _, shorts_rec, _ = analyze_shorts_vs_long(channel_df)
                    fastest_category, _ = get_fastest_growing_category(channel_df)
                    
                    days_active = (channel_df["Publish Time"].max() - channel_df["Publish Time"].min()).days + 1
                    upload_freq_calc = len(channel_df) / days_active if days_active > 0 else 0
                    
                    channel_subs = next((item['Subscribers'] for item in subscriber_data if item['Channel'] == channel_name), 0)
                    
                    comparison_data.append({
                        "Channel": channel_name,
                        "Subscribers": channel_subs,
                        "Total Videos": len(channel_df),
                        "Avg Views": int(channel_df["Views"].mean()),
                        "Avg Engagement (%)": round(channel_df["Engagement Rate (%)"].mean(), 2),
                        "Best Upload Hour": f"{best_hour}:00" if best_hour != "Not enough data" else "N/A",
                        "Best Upload Day": best_day if best_day != "Not enough data" else "N/A",
                        "Content Strategy": shorts_rec.split(" - ")[0] if " - " in shorts_rec else shorts_rec,
                        "Top Category": fastest_category if fastest_category != "Not enough data" else "Mixed",
                        "Upload Frequency": f"{upload_freq_calc:.2f} videos/day",
                        "Total Views": channel_df["Views"].sum()
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                st.markdown("### 📊 Channel Comparison Overview")
                st.dataframe(comparison_df.style.format({
                    "Subscribers": "{:,}",
                    "Avg Views": "{:,}",
                    "Total Views": "{:,}"
                }))
                
                # Enhanced Comparison Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Subscribers vs Performance")
                    fig_scatter = px.scatter(comparison_df, x="Subscribers", y="Avg Views", 
                                           size="Total Videos", color="Avg Engagement (%)",
                                           hover_name="Channel", 
                                           title="📊 Subscribers vs Performance",
                                           labels={"Subscribers": "Subscriber Count", "Avg Views": "Average Views"})
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                   pass
            
                st.markdown("# 🎯 Executive Summary &amp; Action Plan")
        
                st.markdown("### 🏁 Competition Insights")
                
                # Find the leader in each category
                leader_subs = comparison_df.loc[comparison_df["Subscribers"].idxmax(), "Channel"]
                leader_views = comparison_df.loc[comparison_df["Avg Views"].idxmax(), "Channel"]
                leader_engagement = comparison_df.loc[comparison_df["Avg Engagement (%)"].idxmax(), "Channel"]
                most_active = comparison_df.loc[comparison_df["Total Videos"].idxmax(), "Channel"]
                
                col1_sum, col2_sum, col3_sum, col4_sum = st.columns(4)
                with col1_sum:
                    st.success(f"👑 **Subscriber Leader:** {leader_subs}")
                with col2_sum:
                    st.success(f"🏆 **View Leader:** {leader_views}")
                with col3_sum:
                    st.success(f"💝 **Engagement Leader:** {leader_engagement}")
                with col4_sum:
                    st.success(f"⚡ **Most Active:** {most_active}")
                
                # Strategic recommendations for each channel
                st.markdown("### 🎯 Strategic Recommendations for Each Channel")
                
                for _, row in comparison_df.iterrows():
                    channel = row["Channel"]
                    st.markdown(f"#### 📺 {channel}")
                    
                    rank_views = comparison_df["Avg Views"].rank(ascending=False)[comparison_df["Channel"] == channel].iloc[0]
                    rank_engagement = comparison_df["Avg Engagement (%)"].rank(ascending=False)[comparison_df["Channel"] == channel].iloc[0]
                    rank_subs = comparison_df["Subscribers"].rank(ascending=False)[comparison_df["Channel"] == channel].iloc[0]
                    
                    if rank_views == 1:
                        st.markdown("🏆 **Market Position:** View leader - maintain dominance")
                    elif rank_views == len(comparison_df):
                        st.markdown("⚡ **Growth Opportunity:** Focus on increasing reach and views")
                    else:
                        st.markdown(f"📈 **Market Position:** Ranked #{int(rank_views)} in views - room for growth")
                    
                    if rank_engagement == 1:
                        st.markdown("- **Strength:** High engagement - leverage this for growth")
                    if rank_subs == 1:
                        st.markdown("- **Advantage:** Largest subscriber base - focus on retention")
                    if "Focus more on Shorts" in row["Content Strategy"]:
                        st.markdown("- **Strategy:** Double down on Shorts content")
                    elif "Long-form content performs better" in row["Content Strategy"]:
                        st.markdown("- **Strategy:** Invest in quality long-form content")
                    
                    st.markdown(f"- **Optimal Timing:** Upload on {row['Best Upload Day']} at {row['Best Upload Hour']}")
                    st.markdown(f"- **Content Focus:** Emphasize {row['Top Category']} content")

            # Calculate viral scores
            now = pd.Timestamp.now(tz=final_df["Publish Time"].dt.tz)
            final_df["Days Since Upload"] = (now - final_df["Publish Time"]).dt.days + 1
            final_df["Views Per Day"] = final_df["Views"] / final_df["Days Since Upload"]
            
            # Clean data for scaling
            final_df["Views Per Day"].replace([float("inf"), -float("inf")], 0, inplace=True)
            final_df["Engagement Rate (%)"].replace([float("inf"), -float("inf")], 0, inplace=True)
            final_df["Views Per Day"].fillna(0, inplace=True)
            final_df["Engagement Rate (%)"].fillna(0, inplace=True)
            
            # Normalize metrics
            scaler = MinMaxScaler()
            final_df[["Norm Views Per Day", "Norm Engagement"]] = scaler.fit_transform(
                final_df[["Views Per Day", "Engagement Rate (%)"]]
            )
            
            # Calculate viral score
            final_df["Viral Score"] = (
                final_df["Norm Views Per Day"] * 30 +
                final_df["Norm Engagement"] * 25 +
                final_df["Clickbait Title"].astype(int) * 10 +
                final_df["isShort"].astype(int) * 15 +
                final_df["Has Trending Keywords"].astype(int) * 20
            ).round(1)
            
            def viral_label(score):
                if score >= 70:
                    return "🟢 Viral"
                elif score >= 40:
                    return "🟡 Moderate"
                else:
                    return "🔴 Low"
            
            final_df["Viral Label"] = final_df["Viral Score"].apply(viral_label)
            
            # Top viral videos across all channels
            st.markdown("### 🔥 Top 10 Viral Videos Across All Channels")
            top_viral = final_df.sort_values("Viral Score", ascending=False).head(10)
            viral_display = top_viral[["Title", "Channel", "Views", "Engagement Rate (%)", "Content Category", "Viral Score", "Viral Label"]].copy()
            st.dataframe(viral_display.style.format({"Views": "{:,}", "Engagement Rate (%)": "{:.2f}"}))
            
            # SENTIMENT ANALYSIS
            st.markdown("## 🎭 Audience Sentiment Analysis")
            
            def fetch_comments(video_id, api_key, max_results=50):
                youtube = build("youtube", "v3", developerKey=api_key)
                comments = []
                try:
                    response = youtube.commentThreads().list(
                        part="snippet", videoId=video_id,
                        maxResults=max_results, textFormat="plainText"
                    ).execute()
                    for item in response.get("items", []):
                        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                        comments.append(comment)
                except:
                    pass
                return comments

            def analyze_sentiment(comment):
                polarity = TextBlob(comment).sentiment.polarity
                if polarity > 0.1:
                    return "Positive"
                elif polarity < -0.1:
                    return "Negative"
                else:
                    return "Neutral"

            sentiment_results = []
            for channel in final_df["Channel"].unique():
                top_vids = final_df[final_df["Channel"] == channel].sort_values("Views", ascending=False).head(3)
                all_comments = []
                for _, row in top_vids.iterrows():
                    vid_id = row["Video ID"]
                    comments = fetch_comments(vid_id, api_key)
                    all_comments.extend(comments)

                if all_comments:
                    sentiments = [analyze_sentiment(c) for c in all_comments]
                    counts = Counter(sentiments)
                    total = sum(counts.values())
                    pos = round((counts.get("Positive", 0) / total) * 100, 2) if total else 0
                    neu = round((counts.get("Neutral", 0) / total) * 100, 2) if total else 0
                    neg = round((counts.get("Negative", 0) / total) * 100, 2) if total else 0
                    
                    loyalty_score = "High" if pos > 60 else "Medium" if pos > 40 else "Low"

                    sentiment_results.append({
                        "Channel": channel,
                        "Total Comments": total,
                        "Positive %": pos,
                        "Neutral %": neu,
                        "Negative %": neg,
                        "Loyalty Score": loyalty_score
                    })

            if sentiment_results:
                df_sentiment = pd.DataFrame(sentiment_results)
                st.dataframe(df_sentiment)

                fig_sent = px.bar(df_sentiment, x="Channel", y=["Positive %", "Neutral %", "Negative %"],
                                title="🎭 Audience Sentiment Comparison", barmode="group")
                st.plotly_chart(fig_sent, use_container_width=True)
            
                st.markdown("### ✅ Interpretation")
                st.markdown("""
                - Channels with higher **positive sentiment** generally show **higher engagement and loyalty**.
                - A **loyalty score** of *High* indicates a strong community – creators should nurture it with consistent content and audience interaction.
                - A **low loyalty score** might reflect poor content reception, frequent churn, or unaddressed criticism.
                """)
            
            # TOP PERFORMING WORDS ANALYSIS
            st.markdown("## 🔍 Top Performing Keywords Analysis")
            
            stop_words = set([
                'this', 'sir', 'batch', 'that', 'your', 'you', 'are', 'how', 'why', 'what', 'can', 'will',
                'our', 'the', 'i', 'who', 'get', 'all', 'his', 'her', 'one', 'day', 'make',
                'use', 'way', 'show', 'was', 'has', 'with', 'when', 'now', 'and', 'but',
                'or', 'run', 'year', 'using', 'build', 'from', 'for', 'full', 'dry', 'out', 
                'more', 'new', 'video', 'watch'
            ])

            word_view_data = []
            for _, row in final_df.iterrows():
                channel = row["Channel"]
                title = row["Title"].lower()
                views = row["Views"]
                words = re.findall(r'\b\w+\b', title)

                for word in words:
                    if len(word) > 2 and word not in stop_words and not word.isnumeric():
                        word_view_data.append((channel, word, views))

            if word_view_data:
                word_df = pd.DataFrame(word_view_data, columns=["Channel", "Word", "Views"])
                avg_views_df = word_df.groupby(["Channel", "Word"])["Views"].mean().reset_index()
                word_freq_df = word_df.groupby(["Channel", "Word"]).size().reset_index(name="Count")

                top_words_df = avg_views_df.merge(word_freq_df, on=["Channel", "Word"])
                top_words_df = top_words_df[top_words_df["Count"] >= 2]
                top_words_df["Label"] = top_words_df["Word"]
                top_words_df = top_words_df.sort_values(["Channel", "Views"], ascending=[True, False])

                for ch in top_words_df["Channel"].unique():
                    st.markdown(f"### 📝 Top Performing Keywords for {ch}")
                    top10 = top_words_df[top_words_df["Channel"] == ch].head(10)
                    if not top10.empty:
                        fig_word = px.bar(top10, x="Label", y="Views", color="Word",
                                        title=f"Top Performing Keywords - {ch}", text_auto=".2s")
                        fig_word.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_word, use_container_width=True)

            # Download functionality
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Complete Analysis", csv, "youtube_comprehensive_analysis_2025.csv", "text/csv")
            
            # Export individual channel reports
            if st.button("📊 Generate Individual Channel Reports"):
                for channel_name in final_df["Channel"].unique():
                    channel_df = final_df[final_df["Channel"] == channel_name]
                    channel_csv = channel_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        f"📥 Download {channel_name} Report", 
                        channel_csv, 
                        f"{channel_name.replace(' ', '_')}_analysis_2025.csv", 
                        "text/csv",
                        key=f"download_{channel_name}"
                    )

            st.markdown("---")
            st.success("### 🎉 Analysis Complete!")
