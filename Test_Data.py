import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import zipfile
from typing import List, Optional, Dict
from models import (
    PreACTScore, DistrictTrend, SubgroupPerformance, 
    ReadinessBreakdown, DistrictSummary
)

class PreACTDataAPI:
    BASE_URL = "https://dpi.wi.gov/wisedash/download-files/type?field_wisedash_upload_type_value=PreACT"
    DOWNLOAD_BASE = "https://dpi.wi.gov/sites/default/files/wise/downloads/"
    CACHE_DIR = Path("./cache")
    CACHE_DURATION = timedelta(days=30)
    
    def __init__(self):
        self.CACHE_DIR.mkdir(exist_ok=True)
    
    def find_available_files(self):
        try:
            response = requests.get(self.BASE_URL, timeout=10)
            response.raise_for_status()
            import re
            files = re.findall(r'href="([^"]*preact_secure_statewide_certified[^"]*\.zip)"', response.text)
            return sorted(files, reverse=True)
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def get_data(self, school_year: str = None, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        if school_year:
            cache_file = self.CACHE_DIR / f"preact_{school_year}.csv"
            cache_meta = self.CACHE_DIR / f"preact_{school_year}.meta"
            filename = f"preact_secure_statewide_certified_{school_year}.zip"
            url = f"{self.DOWNLOAD_BASE}{filename}"
        else:
            available = self.find_available_files()
            if not available:
                print("No public data files found")
                cache_file = self.CACHE_DIR / "preact_latest.csv"
                if cache_file.exists():
                    print("Using stale cache")
                    return pd.read_csv(cache_file)
                return None
            url = available[0]
            filename = url.split('/')[-1]
            cache_file = self.CACHE_DIR / f"preact_latest.csv"
            cache_meta = self.CACHE_DIR / f"preact_latest.meta"
        
        if not school_year:
            cache_meta = self.CACHE_DIR / "preact_latest.meta"
        
        if not force_refresh and cache_file.exists():
            if cache_meta.exists():
                with open(cache_meta, 'r') as f:
                    meta = json.load(f)
                    cached_date = datetime.fromisoformat(meta['cached_at'])
                    if datetime.now() - cached_date < self.CACHE_DURATION:
                        print(f"Using cached data from {meta.get('filename', 'cache')}")
                        return pd.read_csv(cache_file)
        
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        
        print(f"Downloading {filename}...")
        
        try:
            response = session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            zip_path = self.CACHE_DIR / filename
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            with zipfile.ZipFile(zip_path, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if csv_files:
                    with z.open(csv_files[0]) as csv_file:
                        df = pd.read_csv(csv_file)
                        df.to_csv(cache_file, index=False)
                        
                        with open(cache_meta, 'w') as f:
                            json.dump({
                                'cached_at': datetime.now().isoformat(),
                                'source_url': url,
                                'record_count': len(df),
                                'filename': filename
                            }, f)
                        
                        print(f"Downloaded {len(df)} records from {filename}")
                        return df
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error {e.response.status_code}: {url}")
        except Exception as e:
            print(f"Error: {e}")
        
        if cache_file.exists():
            print("Using stale cache")
            return pd.read_csv(cache_file)
        
        return None
    
    def get_multi_year_data(self, start_year: int = None, end_year: int = None) -> Optional[pd.DataFrame]:
        if start_year is None or end_year is None:
            print("Auto-detecting available data...")
            df = self.get_data()
            return df
        
        all_data = []
        
        for year in range(start_year, end_year + 1):
            school_year = f"{year}-{str(year+1)[-2:]}"
            df = self.get_data(school_year)
            if df is not None:
                all_data.append(df)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"Total records loaded: {len(combined)}")
            return combined
        
        return None
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert to float, handling '*' and other non-numeric values"""
        if pd.isna(value) or value == '*' or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value) -> Optional[int]:
        """Safely convert to int"""
        if pd.isna(value) or value == '*' or value == '':
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def parse_records_to_models(self, df: pd.DataFrame) -> List[PreACTScore]:
        """Convert DataFrame to PreACTScore models"""
        records = []
        
        for _, row in df.iterrows():
            try:
                record = PreACTScore(
                    school_year=str(row['SCHOOL_YEAR']),
                    district_code=str(row['DISTRICT_CODE']),
                    district_name=str(row['DISTRICT_NAME']),
                    school_code=str(row['SCHOOL_CODE']) if pd.notna(row['SCHOOL_CODE']) else None,
                    school_name=str(row['SCHOOL_NAME']) if pd.notna(row['SCHOOL_NAME']) else None,
                    grade_level=int(row['GRADE_LEVEL']),
                    test_subject=str(row['TEST_SUBJECT']),
                    test_group=str(row['TEST_GROUP']),
                    group_by=str(row['GROUP_BY']),
                    group_by_value=str(row['GROUP_BY_VALUE']),
                    student_count=self._safe_int(row['STUDENT_COUNT']),
                    group_count=self._safe_int(row['GROUP_COUNT']),
                    average_score=self._safe_float(row['AVERAGE_SCORE']),
                    test_result=str(row['TEST_RESULT']) if pd.notna(row['TEST_RESULT']) else None,
                    test_result_code=str(row['TEST_RESULT_CODE']) if pd.notna(row['TEST_RESULT_CODE']) else None,
                    charter_indicator=str(row['CHARTER_IND']).lower() == 'yes'
                )
                records.append(record)
            except Exception as e:
                print(f"Warning: Skipping invalid record: {e}")
                continue
        
        return records
    
    def get_district_trends(self, df: pd.DataFrame, district_code: str) -> List[DistrictTrend]:
        district_code_int = int(district_code)
        filtered = df[
            (df['DISTRICT_CODE'] == district_code_int) &
            (df['SCHOOL_CODE'].isna() | (df['SCHOOL_CODE'] == '') | (df['SCHOOL_CODE'] == 0)) &
            (df['GROUP_BY'] == 'All Students') &
            (df['TEST_GROUP'] == 'PreACT')
        ].copy()
        
        if filtered.empty:
            return []
        
        trends = []
        for (year, grade), group in filtered.groupby(['SCHOOL_YEAR', 'GRADE_LEVEL']):
            scores = {}
            for _, row in group.iterrows():
                subject = row['TEST_SUBJECT']
                score = self._safe_float(row['AVERAGE_SCORE'])
                if score is not None:
                    scores[subject.lower()] = score
            
            composite_row = group[group['TEST_SUBJECT'] == 'Composite']
            student_count = 0
            if not composite_row.empty:
                student_count = self._safe_int(composite_row.iloc[0]['STUDENT_COUNT']) or 0
            
            trend = DistrictTrend(
                district_code=district_code,
                district_name=str(filtered.iloc[0]['DISTRICT_NAME']),
                school_year=str(year),
                grade_level=int(grade),
                composite_score=scores.get('composite'),
                english_score=scores.get('english'),
                math_score=scores.get('mathematics'),
                reading_score=scores.get('reading'),
                science_score=scores.get('science'),
                stem_score=scores.get('stem'),
                total_students=student_count
            )
            trends.append(trend)
        
        return sorted(trends, key=lambda x: (x.school_year, x.grade_level))
    
    def get_subgroup_performance(self, df: pd.DataFrame, district_code: str) -> List[SubgroupPerformance]:
        district_code_int = int(district_code)
        filtered = df[
            (df['DISTRICT_CODE'] == district_code_int) &
            (df['SCHOOL_CODE'].isna() | (df['SCHOOL_CODE'] == '') | (df['SCHOOL_CODE'] == 0)) &
            (df['TEST_SUBJECT'] == 'Composite') &
            (df['GROUP_BY'] != 'All Students') &
            (df['TEST_GROUP'] == 'PreACT')
        ].copy()
        
        if filtered.empty:
            return []
        
        subgroups = []
        for _, row in filtered.iterrows():
            score = self._safe_float(row['AVERAGE_SCORE'])
            student_count = self._safe_int(row['STUDENT_COUNT'])
            
            if score is not None and student_count is not None:
                subgroup = SubgroupPerformance(
                    school_year=str(row['SCHOOL_YEAR']),
                    district_code=district_code,
                    district_name=str(row['DISTRICT_NAME']),
                    grade_level=int(row['GRADE_LEVEL']),
                    subgroup_category=str(row['GROUP_BY']),
                    subgroup_value=str(row['GROUP_BY_VALUE']),
                    composite_score=score,
                    student_count=student_count
                )
                subgroups.append(subgroup)
        
        return sorted(subgroups, key=lambda x: (x.school_year, x.grade_level, x.subgroup_category))
    
    def get_readiness_breakdown(self, df: pd.DataFrame, district_code: str) -> List[ReadinessBreakdown]:
        filtered = df[
            (df['DISTRICT_CODE'] == district_code) &
            (df['SCHOOL_CODE'].isna() | (df['SCHOOL_CODE'] == '')) &
            (df['GROUP_BY'] == 'All Students') &
            (df['TEST_GROUP'] == 'PreACT')
        ].copy()
        
        if filtered.empty:
            return []
        
        breakdowns = []
        for _, row in filtered.iterrows():
            breakdown = ReadinessBreakdown(
                school_year=str(row['SCHOOL_YEAR']),
                district_code=district_code,
                district_name=str(row['DISTRICT_NAME']),
                grade_level=int(row['GRADE_LEVEL']),
                test_subject=str(row['TEST_SUBJECT']),
                test_result=str(row['TEST_RESULT']) if pd.notna(row['TEST_RESULT']) else None,
                student_count=self._safe_int(row['STUDENT_COUNT'])
            )
            breakdowns.append(breakdown)
        
        return sorted(breakdowns, key=lambda x: (x.school_year, x.grade_level, x.test_subject))
    
    def get_all_districts(self, df: pd.DataFrame) -> List[Dict]:
        unique_districts = df[['DISTRICT_CODE', 'DISTRICT_NAME']].drop_duplicates()
        districts = [
            {'code': str(row['DISTRICT_CODE']), 'name': str(row['DISTRICT_NAME'])}
            for _, row in unique_districts.iterrows()
        ]
        return sorted(districts, key=lambda x: x['name'])
    
    def get_district_summary(self, district_code: str, start_year: int = 2020, end_year: int = 2024) -> Optional[DistrictSummary]:
        df = self.get_multi_year_data(start_year, end_year)
        if df is None:
            return None
        
        trends = self.get_district_trends(df, district_code)
        subgroups = self.get_subgroup_performance(df, district_code)
        
        if not trends:
            return None
        
        district_name = trends[0].district_name if trends else "Unknown"
        years = sorted(set(t.school_year for t in trends))
        total_students = sum(t.total_students for t in trends if t.total_students)
        
        return DistrictSummary(
            district_code=district_code,
            district_name=district_name,
            years_available=years,
            trends=trends,
            subgroup_performance=subgroups,
            total_students_tested=total_students
        )
    
    def export_to_json(self, district_code: str, output_file: str, start_year: int, end_year: int):
        summary = self.get_district_summary(district_code, start_year, end_year)
        if summary:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(summary.__dict__, f, indent=2, default=str)
            print(f"Exported to {output_file}")

def main():
    api = PreACTDataAPI()
    
    print("=" * 60)
    print("Waukesha District - Auto Data Pull")
    print("=" * 60)
    
    summary = api.get_district_summary("6174", start_year=2020, end_year=2024)
    
    if summary and summary.trends:
        print(f"\nDistrict: {summary.district_name}")
        print(f"Years Available: {', '.join(summary.years_available)}")
        print(f"Total Students Tested: {summary.total_students_tested}")
        
        print(f"\nTrends ({len(summary.trends)} records):")
        for trend in summary.trends[:5]:
            print(f"  {trend.school_year} Grade {trend.grade_level}: Composite={trend.composite_score}")
    else:
        print("No data available for this district")
    
    api.export_to_json("6174", "./output/waukesha_preact.json", 2020, 2024)

if __name__ == "__main__":
    main()
