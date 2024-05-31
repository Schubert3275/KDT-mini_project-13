### 모듈 로딩 -----------------------------------
from flask import Flask


### 어플리케이션 팩토리 함수 --------------------
def create_app():
    # Flask Web Server App 인스턴스 생성
    app = Flask(__name__)

    # 블루프린트
    from .views import main_views

    app.register_blueprint(main_views.bp)

    # Flask Server 인스턴스 반환
    return app
