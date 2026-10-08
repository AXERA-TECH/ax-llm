#pragma once

#include <cstdio>
#include <string>

// Expands a per-layer model file name template such as "qwen3_p128_l%d_together.axmodel".
//
// The template comes from config.json, i.e. from whoever packaged the model, so it is never
// handed to printf as a format string (CWE-134). It must contain exactly one integer
// conversion — %d, %i or %u, optionally with '0'/'-' flags and a width up to 32 (e.g. %02d) —
// plus any number of literal "%%". Everything else is rejected with a reason in `err`.
inline bool expand_layer_template(const std::string &tmpl, int index, std::string &out, std::string *err = nullptr)
{
    auto fail = [&](const std::string &why) {
        if (err) *err = why;
        return false;
    };
    out.clear();
    out.reserve(tmpl.size() + 8);
    int conversions = 0;
    for (size_t i = 0; i < tmpl.size(); ++i)
    {
        const char c = tmpl[i];
        if (c != '%')
        {
            out.push_back(c);
            continue;
        }
        if (i + 1 < tmpl.size() && tmpl[i + 1] == '%')
        {
            out.push_back('%');
            ++i;
            continue;
        }
        // %[flags][width]conv
        size_t j = i + 1;
        std::string flags;
        while (j < tmpl.size() && (tmpl[j] == '0' || tmpl[j] == '-'))
        {
            if (flags.find(tmpl[j]) == std::string::npos) flags.push_back(tmpl[j]);
            ++j;
        }
        int width = 0, width_digits = 0;
        while (j < tmpl.size() && tmpl[j] >= '0' && tmpl[j] <= '9')
        {
            width = width * 10 + (tmpl[j] - '0');
            if (++width_digits > 2 || width > 32) return fail("width in '" + tmpl + "' is too large (max 32)");
            ++j;
        }
        if (j >= tmpl.size() || (tmpl[j] != 'd' && tmpl[j] != 'i' && tmpl[j] != 'u'))
            return fail("unsupported '%' sequence in '" + tmpl + "' (only one %d / %i / %u is allowed)");
        if (++conversions > 1) return fail("more than one layer-index placeholder in '" + tmpl + "'");

        char fmt[16];
        std::snprintf(fmt, sizeof(fmt), "%%%s%d%c", flags.c_str(), width, tmpl[j] == 'u' ? 'u' : 'd');
        char num[48];
        if (tmpl[j] == 'u')
            std::snprintf(num, sizeof(num), fmt, (unsigned)index);
        else
            std::snprintf(num, sizeof(num), fmt, index);
        out += num;
        i = j;
    }
    if (conversions != 1) return fail("no layer-index placeholder (%d) in '" + tmpl + "'");
    return true;
}
