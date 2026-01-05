# =============================================================================
# test_preprocess.py - Tests for Privacy-Safe Preprocessing
# =============================================================================
"""
Tests to ensure PII (Personally Identifiable Information) is properly removed.

These tests are CRITICAL for privacy compliance:
1. Customer names must be removed
2. Customer emails must be removed
3. No emails should appear in the embedded text

Run with: pytest tests/test_preprocess.py -v
"""

import pytest
import pandas as pd
import re
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import (
    drop_pii_columns,
    create_document_text,
    verify_no_emails_in_text,
    normalize_text,
    preprocess_tickets,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_raw_data():
    """Create sample raw data that mimics the Kaggle dataset."""
    return pd.DataFrame({
        "Ticket ID": [1, 2, 3],
        "Customer Name": ["John Doe", "Jane Smith", "Bob Wilson"],
        "Customer Email": ["john@example.com", "jane@test.org", "bob@company.net"],
        "Customer Age": [25, 35, 45],
        "Customer Gender": ["Male", "Female", "Male"],
        "Product Purchased": ["Smart TV", "Laptop", "Phone"],
        "Date of Purchase": ["2023-01-01", "2023-02-15", "2023-03-20"],
        "Ticket Type": ["Technical", "Billing", "Refund"],
        "Ticket Subject": ["TV not working", "Wrong charge", "Want refund"],
        "Ticket Description": [
            "My TV won't turn on after update.",
            "I was charged twice for my order.",
            "Product arrived damaged, need refund.",
        ],
        "Ticket Status": ["Open", "Closed", "Pending"],
        "Resolution": [None, "Refunded duplicate charge", "Processing refund"],
        "Ticket Priority": ["High", "Medium", "High"],
        "Ticket Channel": ["Email", "Chat", "Phone"],
        "First Response Time": ["2h", "1h", "30m"],
        "Time to Resolution": [None, "24h", None],
        "Customer Satisfaction Rating": [None, 4.0, None],
    })


@pytest.fixture
def sample_processed_data(sample_raw_data):
    """Create sample processed data (after PII removal)."""
    return preprocess_tickets(sample_raw_data)


# =============================================================================
# PII REMOVAL TESTS
# =============================================================================

class TestPIIRemoval:
    """Tests for PII column removal."""
    
    def test_customer_name_removed(self, sample_raw_data):
        """Customer Name column must be completely removed."""
        df_clean = drop_pii_columns(sample_raw_data)
        
        assert "Customer Name" not in df_clean.columns, \
            "Customer Name column should be removed!"
    
    def test_customer_email_removed(self, sample_raw_data):
        """Customer Email column must be completely removed."""
        df_clean = drop_pii_columns(sample_raw_data)
        
        assert "Customer Email" not in df_clean.columns, \
            "Customer Email column should be removed!"
    
    def test_ticket_id_preserved(self, sample_raw_data):
        """Ticket ID should be preserved for traceability."""
        df_clean = drop_pii_columns(sample_raw_data)
        
        assert "Ticket ID" in df_clean.columns, \
            "Ticket ID should be preserved!"
    
    def test_other_columns_preserved(self, sample_raw_data):
        """Non-PII columns should be preserved."""
        df_clean = drop_pii_columns(sample_raw_data)
        
        expected_columns = [
            "Product Purchased",
            "Ticket Type",
            "Ticket Subject",
            "Ticket Description",
        ]
        
        for col in expected_columns:
            assert col in df_clean.columns, f"{col} should be preserved!"


# =============================================================================
# DOCUMENT TEXT TESTS
# =============================================================================

class TestDocumentText:
    """Tests for document_text field creation."""
    
    def test_document_text_created(self, sample_raw_data):
        """document_text column should be created."""
        df = drop_pii_columns(sample_raw_data)
        df = create_document_text(df)
        
        assert "document_text" in df.columns, \
            "document_text column should be created!"
    
    def test_document_text_contains_subject(self, sample_raw_data):
        """document_text should contain the ticket subject."""
        df = drop_pii_columns(sample_raw_data)
        df = create_document_text(df)
        
        # Check first row
        assert "TV not working" in df.iloc[0]["document_text"], \
            "document_text should contain ticket subject!"
    
    def test_document_text_contains_description(self, sample_raw_data):
        """document_text should contain the ticket description."""
        df = drop_pii_columns(sample_raw_data)
        df = create_document_text(df)
        
        # Check first row
        assert "TV won't turn on" in df.iloc[0]["document_text"], \
            "document_text should contain ticket description!"
    
    def test_document_text_no_customer_name(self, sample_raw_data):
        """document_text must NOT contain customer names."""
        df = preprocess_tickets(sample_raw_data)
        
        # Check all rows
        for idx, row in df.iterrows():
            text = row["document_text"].lower()
            assert "john doe" not in text, "Customer name found in document_text!"
            assert "jane smith" not in text, "Customer name found in document_text!"
            assert "bob wilson" not in text, "Customer name found in document_text!"
    
    def test_document_text_no_emails(self, sample_raw_data):
        """document_text must NOT contain email addresses."""
        df = preprocess_tickets(sample_raw_data)
        
        # Use regex to find any email patterns
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        
        for idx, row in df.iterrows():
            text = row["document_text"]
            emails_found = re.findall(email_pattern, text)
            assert len(emails_found) == 0, \
                f"Email found in document_text: {emails_found}"


# =============================================================================
# EMAIL VERIFICATION TESTS
# =============================================================================

class TestEmailVerification:
    """Tests for email detection in text."""
    
    def test_verify_no_emails_clean_data(self, sample_processed_data):
        """verify_no_emails_in_text should return True for clean data."""
        is_safe = verify_no_emails_in_text(sample_processed_data)
        assert is_safe, "Clean data should pass email verification!"
    
    def test_verify_no_emails_detects_emails(self):
        """verify_no_emails_in_text should detect emails."""
        # Create data with an email in the text
        df_with_email = pd.DataFrame({
            "document_text": [
                "Please contact john@example.com for help.",
                "Normal text without email.",
            ]
        })
        
        is_safe = verify_no_emails_in_text(df_with_email)
        assert not is_safe, "Should detect email in text!"


# =============================================================================
# TEXT NORMALIZATION TESTS
# =============================================================================

class TestTextNormalization:
    """Tests for text normalization."""
    
    def test_strip_whitespace(self):
        """Should strip leading/trailing whitespace."""
        result = normalize_text("  hello world  ")
        assert result == "hello world"
    
    def test_normalize_multiple_spaces(self):
        """Should normalize multiple spaces to single space."""
        result = normalize_text("hello    world")
        assert result == "hello world"
    
    def test_normalize_multiple_newlines(self):
        """Should normalize 3+ newlines to double newline."""
        result = normalize_text("hello\n\n\n\nworld")
        assert result == "hello\n\nworld"
    
    def test_handle_non_string(self):
        """Should handle non-string input gracefully."""
        result = normalize_text(None)
        assert result == ""
        
        result = normalize_text(123)
        assert result == ""


# =============================================================================
# FULL PIPELINE TESTS
# =============================================================================

class TestFullPipeline:
    """Tests for the complete preprocessing pipeline."""
    
    def test_preprocess_returns_dataframe(self, sample_raw_data):
        """preprocess_tickets should return a DataFrame."""
        result = preprocess_tickets(sample_raw_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_preprocess_row_count_preserved(self, sample_raw_data):
        """Row count should be preserved after preprocessing."""
        result = preprocess_tickets(sample_raw_data)
        assert len(result) == len(sample_raw_data)
    
    def test_preprocess_creates_document_text(self, sample_raw_data):
        """Preprocessing should create document_text column."""
        result = preprocess_tickets(sample_raw_data)
        assert "document_text" in result.columns


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
