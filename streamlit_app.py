import streamlit as st
import pandas as pd
import re
import plotly.express as px
from datetime import datetime
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Line Chat Analyzer", layout="wide")
st.title("Line Chat Analyzer")

def parse_line_chat(text):
    # Initialize data containers
    messages = []
    
    # Check format
    is_android = "Chat history with" in text[:100] and not "[LINE]" in text[:100]
    
    # Display format detected
    if is_android:
        st.info("Detected Android format Line chat")
    else:
        st.info("Detected iPhone format Line chat")
    
    # Regex patterns
    iphone_date_pattern = r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{1,2}/\d{1,2}/\d{4}) BE'
    android_date_pattern = r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{1,2}/\d{1,2}/\d{4})'
    message_pattern = r'(\d{2}:\d{2})\t([^\t]*)\t(.+)'
    
    # Process each line
    current_date = None
    
    for line in text.strip().split('\n'):
        if not line.strip():
            continue
           
        # Check for date in Android format
        if is_android:
            date_match = re.match(android_date_pattern, line)
            if date_match:
                try:
                    _, date_str = date_match.groups()
                    month, day, year = map(int, date_str.split('/'))
                    current_date = f"{day:02d}/{month:02d}/{year:04d}"
                    continue
                except:
                    pass
        # Check for date in iPhone format
        else:
            date_match = re.match(iphone_date_pattern, line)
            if date_match:
                try:
                    _, date_str = date_match.groups()
                    day, month, year = map(int, date_str.split('/'))
                    
                    # Convert Buddhist Era to Common Era
                    if year > 2500:
                        year -= 543
                        
                    current_date = f"{day:02d}/{month:02d}/{year:04d}"
                    continue
                except:
                    pass
            
        if current_date:
            message_match = re.match(message_pattern, line)
            if message_match:
                time, sender, content = message_match.groups()
                
                # Determine message type
                msg_type = 'text'
                if '[Sticker]' in content:
                    msg_type = 'sticker'
                elif '[Photo]' in content:
                    msg_type = 'photo'
                elif '[Video]' in content:
                    msg_type = 'video'
                elif '[File]' in content:
                    msg_type = 'file'
                elif '☎' in content:
                    if 'Call time' in content:
                        msg_type = 'call'
                    else:
                        msg_type = 'missed_call'
                elif 'unsent a message' in content:
                    msg_type = 'unsent'
                
                # Add message to list
                timestamp = f"{current_date} {time}"
                messages.append({
                    'timestamp': timestamp,
                    'sender': sender,
                    'content': content,
                    'type': msg_type
                })
    
    # Create DataFrame
    if not messages:
        return pd.DataFrame()
        
    df = pd.DataFrame(messages)
    
    # Drop senders with empty or whitespace-only names
    df = df[df['sender'].str.strip() != '']
    
    # Convert timestamp to datetime
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M', dayfirst=True)
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()
        df['message_length'] = df['content'].apply(lambda x: len(str(x)))
        
        # Detect session boundaries (>30 min gap = new session)
        df = df.sort_values('timestamp')
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60
        df['session_id'] = (df['time_diff'] > 30).cumsum()
        
        # Add conversational turn tracking
        df['prev_sender'] = df['sender'].shift(1)
        df['conversational_turn'] = df['sender'] != df['prev_sender']
        df['session_turn_count'] = df.groupby('session_id')['conversational_turn'].cumsum()
    except Exception as e:
        st.error(f"Error processing timestamps: {str(e)}")
        st.write("Sample data causing the error:")
        st.write(df['timestamp'].head())
    
    return df

def analyze_chat_sessions(df):
    # Ensure hour column exists
    df['hour_of_day'] = df['hour']
    
    # Create session summary
    session_summary = df.groupby('session_id').agg(
        initiator=('sender', 'first'),
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        duration_minutes=('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60),
        message_count=('content', 'count'),
        turn_count=('conversational_turn', 'sum'),
        hour_of_day=('hour_of_day', lambda x: x.min()),
        weekday=('day', 'first')
    ).reset_index()
    
    return session_summary

# File uploader
uploaded_file = st.file_uploader("Upload your Line chat export file (.txt)", type="txt")

if uploaded_file:
    # Read and process the file
    with st.spinner("Analyzing chat..."):
        try:
            chat_content = uploaded_file.read().decode('utf-8')
            df = parse_line_chat(chat_content)
            
            if df.empty:
                st.error("Couldn't parse any messages. Check your file format.")
            else:
                # Show a sample of parsed messages to verify
                st.write("Sample of parsed messages:")
                st.dataframe(df[['timestamp', 'sender', 'content', 'type']].head())
                
                # Get session analysis
                analyze = analyze_chat_sessions(df)
                
                # Display overview stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Messages", len(df))
                with col2:
                    st.metric("Participants", df['sender'].nunique())
                with col3:
                    st.metric("Date Range", f"{df['date'].min()} - {df['date'].max()}")
                
                # Show detected participants
                st.write("### Detected Participants")
                participants = df['sender'].unique()
                st.write(", ".join(participants))
                
                # Tabs for analysis
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Message Stats", "Time Analysis", "Message Types", "ANOVA Analysis", "Yapper Analysis"])
                
                with tab1:
                    # Message distribution
                    sender_counts = df['sender'].value_counts().reset_index()
                    sender_counts.columns = ['Sender', 'Message Count']
                    
                    fig = px.pie(sender_counts, values='Message Count', names='Sender', title='Message Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Message length
                    avg_length = df.groupby('sender')['message_length'].mean().reset_index()
                    fig = px.bar(avg_length, x='sender', y='message_length', title='Average Message Length')
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Activity by hour
                    hour_counts = df.groupby('hour').size().reset_index(name='count')
                    fig = px.bar(hour_counts, x='hour', y='count', title='Messages by Hour')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Activity by day
                    day_counts = df.groupby('day').size().reset_index(name='count')
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    day_counts['day'] = pd.Categorical(day_counts['day'], categories=day_order, ordered=True)
                    day_counts = day_counts.sort_values('day')
                    
                    fig = px.bar(day_counts, x='day', y='count', title='Messages by Day')
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    # Message types
                    type_counts = df['type'].value_counts().reset_index()
                    type_counts.columns = ['Type', 'Count']
                    
                    fig = px.pie(type_counts, values='Count', names='Type', title='Message Types')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Conversation starters
                    starters = df[df['time_diff'].isnull() | (df['time_diff'] > 30)]['sender'].value_counts().reset_index()
                    starters.columns = ['Sender', 'Count']
                    
                    fig = px.bar(starters, x='Sender', y='Count', title='Conversation Initiators')
                    st.plotly_chart(fig, use_container_width=True)
                
                # ANOVA Analysis tab
                with tab4:
                    st.subheader("ANOVA Analysis: Time Periods")
                    
                    # Define bins for 4-hour chunks (0-4, 4-8, 8-12, 12-16, 16-20, 20-24)
                    bins = [0, 4, 8, 12, 16, 20, 24]
                    labels = ['Late Night', 'Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night']
                    
                    # Add time period to analysis dataframe
                    analyze['time_period'] = pd.cut(analyze['hour_of_day'], bins=bins, labels=labels, right=False)
                    
                    # Display time period distribution
                    st.write("Message counts by time period:")
                    period_counts = analyze.groupby('time_period').size().reset_index(name='conversation_count')
                    
                    fig = px.bar(
                        period_counts, 
                        x='time_period', 
                        y='conversation_count',
                        title='Conversation Distribution by Time Period',
                        labels={'time_period': 'Time Period', 'conversation_count': 'Number of Conversations'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Perform ANOVA on turn_count across time periods
                    groups = []
                    valid_labels = []
                    
                    for label in labels:
                        group_data = analyze[analyze['time_period'] == label]['turn_count']
                        if len(group_data) >= 3:  # Need at least 3 data points per group
                            groups.append(group_data)
                            valid_labels.append(label)
                    
                    if len(groups) >= 2:  # Need at least 2 groups for ANOVA
                        try:
                            f_stat, p_value = f_oneway(*groups)
                            
                            # Show ANOVA results
                            st.write("### ANOVA Results")
                            st.write(f"F-statistic: {f_stat:.4f}")
                            st.write(f"p-value: {p_value:.4f}")
                            
                            if p_value < 0.05:
                                st.write("Conclusion: There are significant differences in conversation turn counts between time periods.")
                                
                                # Perform Tukey's HSD test for pairwise comparisons
                                try:
                                    # Create a dataframe for Tukey test
                                    tukey_data = []
                                    for i, label in enumerate(valid_labels):
                                        for val in analyze[analyze['time_period'] == label]['turn_count']:
                                            tukey_data.append({'time_period': label, 'turn_count': val})
                                    
                                    tukey_df = pd.DataFrame(tukey_data)
                                    
                                    # Run Tukey's test
                                    tukey = pairwise_tukeyhsd(
                                        endog=tukey_df['turn_count'],
                                        groups=tukey_df['time_period'],
                                        alpha=0.05
                                    )
                                    
                                    # Display Tukey results
                                    tukey_results = pd.DataFrame(
                                        data=tukey._results_table.data[1:],
                                        columns=tukey._results_table.data[0]
                                    )
                                    
                                    # Only show significant differences
                                    significant_pairs = tukey_results[tukey_results['reject']]
                                    
                                    if not significant_pairs.empty:
                                        st.write("### Significant Differences Between Time Periods")
                                        st.dataframe(significant_pairs)
                                    else:
                                        st.write("No significant pairwise differences found despite overall ANOVA significance.")
                                except Exception as e:
                                    st.write(f"Error performing Tukey's test: {str(e)}")
                            else:
                                st.write("Conclusion: No significant differences in conversation turn counts between time periods.")
                        except Exception as e:
                            st.write(f"Error performing ANOVA: {str(e)}")
                    else:
                        st.write("Not enough data in time periods for ANOVA analysis.")
                    
                    # Display time period summary statistics
                    st.write("### Summary Statistics by Time Period")
                    summary_stats = analyze.groupby('time_period')['turn_count'].agg([
                        'count', 'mean', 'std', 'min', 'max'
                    ]).reset_index()
                    
                    if not summary_stats.empty:
                        # Round numeric columns for display
                        numeric_cols = ['mean', 'std', 'min', 'max']
                        summary_stats[numeric_cols] = summary_stats[numeric_cols].round(2)
                        st.dataframe(summary_stats)
                        
                        # Boxplot visualization
                        st.write("### Turn Count Distribution by Time Period")
                        fig = px.box(
                            analyze, 
                            x='time_period', 
                            y='turn_count',
                            title='Conversation Turn Count Distribution by Time Period'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Not enough data for summary statistics.")
                
                # Yapper Analysis tab
                with tab5:
                    st.subheader("Yapper Analysis: Conversation Dominance")
                    
                    # Step 1: Count unique senders per session
                    unique_speaker_counts = df.groupby('session_id')['sender'].nunique().reset_index()
                    unique_speaker_counts.columns = ['session_id', 'unique_speakers']
                    
                    # Step 2: Filter only multi-speaker sessions
                    multi_speaker_sessions = unique_speaker_counts[unique_speaker_counts['unique_speakers'] > 1]
                    
                    if not multi_speaker_sessions.empty:
                        # Step 3: Keep only conversations with multiple participants
                        conversation_df = df[df['session_id'].isin(multi_speaker_sessions['session_id'])].copy()
                        
                        # Step 4: Count messages per sender per session
                        message_counts = conversation_df.groupby(['session_id', 'sender']).size().reset_index(name='message_count')
                        
                        # Step 5: Count total messages per session
                        total_messages = conversation_df.groupby('session_id').size().reset_index(name='total_messages')
                        
                        # Step 6: Merge message counts with total messages
                        dominance_df = message_counts.merge(total_messages, on='session_id', how='left')
                        
                        # Step 7: Compute dominance ratio
                        dominance_df['dominance_ratio'] = dominance_df['message_count'] / dominance_df['total_messages']
                        
                        # Display dominance df
                        st.write("### Message Dominance by Conversation and Participant")
                        st.dataframe(dominance_df)
                        
                        # Get unique senders
                        senders = dominance_df['sender'].unique()
                        
                        if len(senders) >= 2:
                            # Extract dominance ratios for each participant (first two)
                            sender1 = senders[0]
                            sender2 = senders[1]
                            
                            dominance_person1 = dominance_df[dominance_df['sender'] == sender1]['dominance_ratio']
                            dominance_person2 = dominance_df[dominance_df['sender'] == sender2]['dominance_ratio']
                            
                            # T-test on dominance ratios
                            if len(dominance_person1) > 1 and len(dominance_person2) > 1:
                                t_stat, p_value = ttest_ind(dominance_person1, dominance_person2, equal_var=False)
                                
                                st.write("### Dominance T-Test Results")
                                st.write(f"T-statistic: {t_stat:.4f}")
                                st.write(f"p-value: {p_value:.4f}")
                                
                                if p_value < 0.05:
                                    if dominance_person1.mean() > dominance_person2.mean():
                                        st.write(f"Conclusion: {sender1} is significantly more dominant in conversations.")
                                    else:
                                        st.write(f"Conclusion: {sender2} is significantly more dominant in conversations.")
                                else:
                                    st.write("Conclusion: No significant difference in dominance between participants.")
                            
                            # Visualize dominance comparison
                            st.write("### Dominance Ratio Comparison")
                            
                            # Box plot
                            fig = go.Figure()
                            fig.add_trace(go.Box(y=dominance_person1, name=sender1))
                            fig.add_trace(go.Box(y=dominance_person2, name=sender2))
                            fig.update_layout(
                                title="Dominance Ratio Distribution by Participant",
                                yaxis_title="Dominance Ratio",
                                boxmode='group'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Bar chart - average dominance
                            avg_dominance = dominance_df.groupby('sender')['dominance_ratio'].mean().reset_index()
                            
                            fig = px.bar(
                                avg_dominance,
                                x='sender',
                                y='dominance_ratio',
                                title='Average Dominance Ratio by Participant',
                                labels={'sender': 'Participant', 'dominance_ratio': 'Avg. Dominance Ratio'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("Need at least two participants for dominance analysis.")
                    else:
                        st.write("No multi-speaker conversations found for dominance analysis.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please make sure you're uploading a valid Line chat export file.")
else:
    st.markdown("""
    ## How to use the Line Chat Analyzer
    
    1. **Export your Line chat conversation**:
       - Open the chat in Line
       - Go to chat settings (gear icon)
       - Select "Export chat history"
       - Save as a text file
    
    2. **Upload the file**:
       - Use the file uploader above
       - Wait for the analysis to complete
    
    All data processing happens in your browser - no information is stored on any server.
    """)