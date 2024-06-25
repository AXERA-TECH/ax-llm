#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QTextBrowser>
#include <thread>
#include <QFileDialog>
#include <opencv2/opencv.hpp>

#include "src/runner/utils/sample_log.h"

void llm_running_callback(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve)
{
    MainWindow *p = (MainWindow *)reserve;
    p->append_textedit(p_str);
}

void thread_llm_runing(LLM *p_llm, std::vector<unsigned short> *p_data, MainWindow *p_win)
{
    ALOGI("thread_llm_runing begin");
    p_win->enable_controls(false);
    p_llm->Run(*p_data);
    p_win->enable_controls(true);
    ALOGI("thread_llm_runing end");
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(this, &MainWindow::sig_llm_output, this, &MainWindow::on_llm_output);
    connect(this, &MainWindow::sig_enable_controls, this, &MainWindow::on_enable_controls);
}

MainWindow::~MainWindow()
{
    m_llm.Deinit();
    delete ui;
}

bool MainWindow::InitLLM(LLMAttrType attr)
{
    return m_llm.Init(attr);
}

void MainWindow::append_textedit(const char *p_str)
{
    emit sig_llm_output(QString(p_str));
}

void MainWindow::on_llm_output(QString str)
{
    if (ui->ckbox_chinese.isChecked())
    {
        if (str.endsWith("奖品"))
        {
            m_llm.Stop();
            str = str.replace("奖品", "");
        }
    }
    else
    {
        str += " ";
    }
    ui->txt_out->append(str);
    printf("%s", str.toStdString().c_str());
    fflush(stdout);
}

void MainWindow::enable_controls(bool b_enable)
{
    emit sig_enable_controls(b_enable);
}

void MainWindow::on_enable_controls(bool b_enable)
{
    ui->btn_ask->setEnabled(b_enable);
    ui->ckbox_chinese->setEnabled(b_enable);
}

void MainWindow::on_btn_ask_clicked()
{
    auto filename = QFileDialog::getOpenFileName(this, "", "", "image(*.png *.jpg *.jpeg *.bmp)");
    if (filename.isEmpty())
    {
        return;
    }
    QImage img(filename);
    ui->lab_img->SetImage(img);
    cv::Mat src = cv::imread(filename.toStdString());
    std::vector<unsigned short> out_embed;
    m_llm.RunVpm(src, out_embed);

    if (ui->ckbox_chinese.isChecked())
    {
        memcpy(m_data_zh.data() + 5 * m_llm.getAttr()->tokens_embed_size, out_embed.data(), out_embed.size() * sizeof(unsigned short));

        // 开线程run
        std::thread t(thread_llm_runing, &m_llm, &m_data_zh, this);
        t.detach();
    }
    else
    {
        memcpy(m_data_en.data() + 5 * m_llm.getAttr()->tokens_embed_size, out_embed.data(), out_embed.size() * sizeof(unsigned short));
        // 开线程run
        std::thread t(thread_llm_runing, &m_llm, &m_data_en, this);
        t.detach();
    }
}

void MainWindow::on_btn_stop_clicked()
{
    m_llm.Stop();
}
