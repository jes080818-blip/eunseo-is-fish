import requests
from pathlib import Path
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import urllib.request

def download_google_images_selenium(search_query, save_folder, num_images=100):
    """
    Selenium을 사용한 구글 이미지 크롤링
    
    Args:
        search_query: 검색어 (예: "dirty dishes", "clean dishes")
        save_folder: 저장할 폴더 경로
        num_images: 다운로드할 이미지 개수
    """
    # 저장 폴더 생성
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Chrome 옵션 설정
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 브라우저 창 안 띄우기
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # 드라이버 실행
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # 구글 이미지 검색
        search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
        driver.get(search_url)
        time.sleep(2)
        
        downloaded = 0
        scroll_count = 0
        
        while downloaded < num_images:
            # 페이지 스크롤
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # 썸네일 이미지 찾기
            thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.rg_i")
            
            for img in thumbnails[downloaded:]:
                if downloaded >= num_images:
                    break
                
                try:
                    # 썸네일 클릭
                    img.click()
                    time.sleep(1)
                    
                    # 큰 이미지 찾기
                    large_images = driver.find_elements(By.CSS_SELECTOR, "img.n3VNCb")
                    
                    for large_img in large_images:
                        src = large_img.get_attribute("src")
                        
                        if src and src.startswith("http"):
                            try:
                                # 이미지 다운로드
                                file_path = save_path / f"{search_query}_{downloaded:04d}.jpg"
                                urllib.request.urlretrieve(src, file_path)
                                print(f"다운로드 완료: {downloaded + 1}/{num_images} - {file_path.name}")
                                downloaded += 1
                                break
                            except Exception as e:
                                print(f"다운로드 실패: {e}")
                                continue
                
                except Exception as e:
                    print(f"이미지 처리 실패: {e}")
                    continue
            
            # "결과 더보기" 버튼 클릭
            scroll_count += 1
            if scroll_count % 5 == 0:
                try:
                    more_button = driver.find_element(By.CSS_SELECTOR, ".mye4qd")
                    more_button.click()
                    time.sleep(2)
                except:
                    pass
        
        print(f"\n총 {downloaded}개 이미지 다운로드 완료!")
        
    finally:
        driver.quit()


def download_google_images_simple(search_query, save_folder, num_images=100):
    """
    icrawler 라이브러리 사용 (더 간단한 방법)
    
    설치 필요: pip install icrawler
    """
    from icrawler.builtin import GoogleImageCrawler
    
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    
    google_crawler = GoogleImageCrawler(storage={'root_dir': str(save_path)})
    google_crawler.crawl(
        keyword=search_query,
        max_num=num_images,
        min_size=(200, 200),  # 최소 이미지 크기
    )
    
    print(f"{search_query}: {num_images}개 다운로드 완료!")


# ============ 사용 예시 ============

if __name__ == "__main__":
    # 데이터 저장 경로
    data_root = Path("/HOME/2026_youth/ai26/wash_data")
    
    # 방법 1: icrawler 사용 (추천 - 더 간단함)
    print("=== icrawler로 이미지 다운로드 ===")
    print("\n설치 필요: pip install icrawler")
    print("\n사용법:")
    print("download_google_images_simple('dirty dishes', data_root / 'dirty_crawled', 500)")
    print("download_google_images_simple('clean dishes', data_root / 'clear_crawled', 500)")
    
    # 방법 2: Selenium 사용
    print("\n\n=== Selenium으로 이미지 다운로드 ===")
    print("\n설치 필요: pip install selenium")
    print("Chrome 드라이버 필요: https://chromedriver.chromium.org/")
    print("\n사용법:")
    print("download_google_images_selenium('dirty dishes', data_root / 'dirty_crawled', 500)")
    print("download_google_images_selenium('clean dishes', data_root / 'clear_crawled', 500)")
    
    # 실제 실행 (주석 해제해서 사용)
    # download_google_images_simple('dirty dishes', data_root / 'dirty_crawled', 500)
    # download_google_images_simple('clean dishes', data_root / 'clear_crawled', 500)