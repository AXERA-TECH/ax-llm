#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QTextBrowser>
#include <thread>

#include "src/runner/utils/sample_log.h"

void llm_running_callback(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve)
{
    MainWindow *p = (MainWindow *)reserve;
    p->append_textedit(p_str);
}

void thread_llm_runing(LLM *p_llm, QString *p_str, MainWindow *p_win)
{
    ALOGI("thread_llm_runing begin");
    p_win->enable_controls(false);
    p_llm->Run(p_str->toStdString());
    delete p_str;
    p_win->enable_controls(true);
    ALOGI("thread_llm_runing end");
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_vBoxLayout = new QVBoxLayout();
    ui->widget->setLayout(m_vBoxLayout);
    m_vBoxLayout->addSpacing(1080);
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
    m_textedit_output_vec.back()->append(p_str);
    printf("%s", p_str);
    fflush(stdout);
}

void MainWindow::enable_controls(bool b_enable)
{
    ui->btn_ask->setEnabled(b_enable);
    ui->txt_msg->setEnabled(b_enable);
}

void resize_textedit(QTextEdit *textedit)
{
    int newHeight = textedit->document()->size().height();
    if (newHeight < 60)
        newHeight = 60;
    printf("%d\n", newHeight);
    fflush(stdout);
    textedit->setFixedHeight(newHeight + 30);
}

QTextEdit *create_textedit(QString rgb_color = "220,127,125")
{
    QTextBrowser *textedit = new QTextBrowser;
    textedit->setStyleSheet(QString("border-radius: 20px;background-color: rgb(%1);").arg(rgb_color));
    auto font = textedit->font();
    font.setPointSize(18);
    textedit->setFont(font);
    textedit->setMinimumSize(0, 60);
    textedit->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
    textedit->setReadOnly(true);
    QObject::connect(textedit, &QTextEdit::textChanged, textedit, [textedit]()
                     { resize_textedit(textedit); });
    return textedit;
}

void MainWindow::on_btn_ask_clicked()
{
    auto txtedit_input = create_textedit();

    QHBoxLayout *hboxlayout = new QHBoxLayout();
    hboxlayout->addWidget(txtedit_input);
    hboxlayout->addSpacing(200);

    auto txtedit_output = create_textedit("125,127,220");
    // txtedit_output->setHtml(ui->txt_msg->text());
    QHBoxLayout *hboxlayout1 = new QHBoxLayout();
    hboxlayout1->addSpacing(200);
    hboxlayout1->addWidget(txtedit_output);

    m_vBoxLayout->insertLayout(0, hboxlayout);
    m_vBoxLayout->insertLayout(1, hboxlayout1);

    m_textedit_input_vec.push_back(txtedit_input);
    m_textedit_output_vec.push_back(txtedit_output);
    // resize_textedit(txtedit);
    // resize_textedit(txtedit1);

    txtedit_input->setText(ui->txt_msg->text());

    // 开线程run
    QString *str_msg = new QString(ui->txt_msg->text());
    std::thread t(thread_llm_runing, &m_llm, str_msg, this);
    t.detach();
    ui->txt_msg->clear();
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    for (auto textedit : m_textedit_input_vec)
    {
        resize_textedit(textedit);
    }
    for (auto textedit : m_textedit_output_vec)
    {
        resize_textedit(textedit);
    }

    QWidget::resizeEvent(event);
}

void MainWindow::on_txt_msg_returnPressed()
{
    on_btn_ask_clicked();
}
