# utils/ui_components.py
"""
Enhanced UI Components
Example questions, quick actions, and interactive elements
"""

import streamlit as st
from typing import List, Dict, Optional, Callable


# ExampleQuestions class removed - users can ask any questions freely


class QuickActions:
    """Quick action buttons for common tasks"""
    
    @staticmethod
    def show_actions(
        has_documents: bool = False,
        has_chat_history: bool = False,
        actions_config: Dict = None
    ):
        """
        Display quick action buttons
        
        Args:
            has_documents: Whether documents are loaded
            has_chat_history: Whether chat history exists
            actions_config: Optional custom actions configuration
        """
        # Quick actions section removed - users can interact freely through chat
        # Export functionality moved to Export tab
        return None


class DocumentCard:
    """Displays document information as a card"""
    
    @staticmethod
    def show(
        doc_name: str,
        doc_metadata: Dict,
        is_selected: bool = False,
        show_details: bool = True
    ):
        """
        Display a document card
        
        Args:
            doc_name: Document name
            doc_metadata: Document metadata
            is_selected: Whether document is selected
            show_details: Whether to show detailed metadata
        """
        # Card styling
        border_color = "#4CAF50" if is_selected else "#ddd"
        bg_color = "#f0f8f0" if is_selected else "#fafafa"
        
        card_html = f"""
        <div style="
            border: 2px solid {border_color};
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: {bg_color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="font-weight: bold; font-size: 16px; margin-bottom: 10px; color: #333;">
                {'[✓] ' if is_selected else '[ ] '}{doc_name}
            </div>
        """
        
        if show_details:
            stats = []
            if 'num_pages' in doc_metadata:
                stats.append(f"{doc_metadata['num_pages']} pages")
            if 'num_chunks' in doc_metadata:
                stats.append(f"{doc_metadata['num_chunks']} chunks")
            if 'size_mb' in doc_metadata:
                stats.append(f"{doc_metadata['size_mb']:.1f} MB")
            
            if stats:
                card_html += f"""
                <div style="font-size: 12px; color: #666; margin-top: 8px;">
                    {' • '.join(stats)}
                </div>
                """
        
        card_html += "</div>"
        st.markdown(card_html, unsafe_allow_html=True)


class ProgressIndicator:
    """Animated progress indicators"""
    
    @staticmethod
    def show_processing(message: str = "Processing", step: int = 1, total: int = 1):
        """
        Show animated processing indicator
        
        Args:
            message: Processing message
            step: Current step
            total: Total steps
        """
        html = f"""
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        @keyframes dots {{
            0%, 20% {{ content: '.'; }}
            40% {{ content: '..'; }}
            60%, 100% {{ content: '...'; }}
        }}
        .spinner {{
            display: inline-block;
            animation: spin 1.5s linear infinite;
            font-size: 24px;
        }}
        .dots::after {{
            content: '';
            animation: dots 1.5s steps(3, end) infinite;
        }}
        </style>
        <div style="
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            margin: 20px 0;
        ">
            <div class="spinner">●</div>
            <div style="font-weight: 500; margin-top: 10px;" class="dots">{message}</div>
            <div style="font-size: 12px; margin-top: 8px; opacity: 0.9;">
                Step {step} of {total}
            </div>
            <div style="margin-top: 10px;">
                <div style="background: rgba(255,255,255,0.3); border-radius: 10px; height: 6px;">
                    <div style="
                        background: white;
                        border-radius: 10px;
                        height: 6px;
                        width: {(step/total)*100}%;
                        transition: width 0.3s;
                    "></div>
                </div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    @staticmethod
    def show_success(message: str = "Complete!"):
        """Show success message"""
        html = f"""
        <div style="
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            text-align: center;
            margin: 20px 0;
        ">
            <div style="font-size: 32px; margin-bottom: 10px;">✓</div>
            <div style="font-weight: 500;">{message}</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


class ConfidenceBadge:
    """Display confidence score with visual indicator"""
    
    @staticmethod
    def show(confidence: float, show_details: bool = False):
        """
        Display confidence badge
        
        Args:
            confidence: Confidence score (0-100)
            show_details: Whether to show detailed breakdown
        """
        if confidence >= 70:
            color = "#4CAF50"
            level = "High"
            icon = "●"
        elif confidence >= 50:
            color = "#FF9800"
            level = "Medium"
            icon = "●"
        else:
            color = "#F44336"
            level = "Low"
            icon = "●"
        
        html = f"""
        <div style="
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            background-color: {color}20;
            border: 2px solid {color};
            margin: 10px 0;
        ">
            <span style="font-size: 18px;">{icon}</span>
            <span style="
                font-weight: bold;
                color: {color};
                margin-left: 8px;
                font-size: 16px;
            ">{level} Confidence: {confidence:.1f}%</span>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        
        if show_details:
            st.caption(
                f"This score is based on source relevance, answer completeness, and consistency. "
                f"{'Highly reliable answer.' if confidence >= 70 else 'Consider verifying with source papers.' if confidence >= 50 else 'Low confidence - manual verification recommended.'}"
            )


class SearchModeSelector:
    """Interactive search mode selector"""
    
    @staticmethod
    def show() -> tuple:
        """
        Show search mode selector
        
        Returns:
            (search_mode, alpha) tuple
        """
        search_mode = st.radio(
            "Search Mode",
            ["Semantic", "Keyword", "Hybrid"],
            horizontal=True,
            help="Semantic: meaning-based | Keyword: exact matches | Hybrid: combines both"
        )
        
        alpha = 0.5
        if search_mode == "Hybrid":
            alpha = st.slider(
                "Semantic Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="1.0 = pure semantic, 0.0 = pure keyword"
            )
        elif search_mode == "Semantic":
            alpha = 1.0
        else:  # Keyword
            alpha = 0.0
        
        return search_mode, alpha
