import re

with open('templates/base.html', 'r') as f:
    content = f.read()

# 1. Colors tailwind config
content = content.replace(
"""                        'brand-yellow': {
                            DEFAULT: '#FFCC00',
                            pastel: '#FFFDF0',
                            accent: '#FFD700',
                        },
                        'brand-blue': {
                            DEFAULT: '#00247D',
                            pastel: '#F0F4FF',
                            accent: '#0033B3',
                        },""",
"""                        'brand-yellow': {
                            DEFAULT: '#fde68a',
                            pastel: '#fef3c7',
                            accent: '#f59e0b',
                        },
                        'brand-blue': {
                            DEFAULT: '#64748b',
                            pastel: '#f1f5f9',
                            accent: '#818cf8',
                        },""")

# 2. Dark mode body style
content = content.replace(
"""            .dark body {
                @apply bg-zinc-950 text-zinc-100;
            }""",
"""            .dark body {
                @apply bg-slate-900 text-zinc-100;
            }""")

# 3. Sidebar background and text
content = content.replace(
"""    <aside id="sidebar"
        class="fixed top-0 left-0 z-40 w-64 h-screen transition-transform duration-300 -translate-x-full md:translate-x-0 bg-slate-900 text-slate-300 flex flex-col border-r border-slate-800">""",
"""    <!-- Sidebar -->
    <aside id="sidebar"
        class="fixed top-0 left-0 z-40 w-64 h-screen transition-transform duration-300 -translate-x-full md:translate-x-0 bg-slate-50 dark:bg-slate-900 text-slate-700 dark:text-slate-300 flex flex-col border-r border-slate-200 dark:border-slate-800">""")

# 4. Brand logo
content = content.replace(
"""                        <span class="text-white font-bold text-xl">C</span>
                    </div>
                    <h1 class="text-xl font-bold tracking-tight text-white">CV <span
                            class="font-light text-slate-400">College</span></h1>""",
"""                        <span class="text-brand-blue dark:text-white font-bold text-xl">C</span>
                    </div>
                    <h1 class="text-xl font-bold tracking-tight text-slate-800 dark:text-white">CV <span
                            class="font-light text-slate-500 dark:text-slate-400">College</span></h1>""")

# 5. Profile card
content = content.replace(
"""                    <div class="flex items-center gap-3 bg-slate-800/40 p-3 rounded-xl border border-slate-800">
                        <div class="w-10 h-10 rounded-full bg-slate-700 flex items-center justify-center text-white font-bold shrink-0 border border-slate-600">
                            {{ request.user.first_name|first|upper }}{{ request.user.last_name|first|upper }}
                        </div>
                        <div class="overflow-hidden">
                            <p class="text-sm font-semibold text-white truncate">{{ request.user.get_full_name|default:request.user.username }}</p>""",
"""                    <div class="flex items-center gap-3 bg-slate-100 dark:bg-slate-800/40 p-3 rounded-xl border border-slate-200 dark:border-slate-800">
                        <div class="w-10 h-10 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center text-slate-700 dark:text-white font-bold shrink-0 border border-slate-300 dark:border-slate-600">
                            {{ request.user.first_name|first|upper }}{{ request.user.last_name|first|upper }}
                        </div>
                        <div class="overflow-hidden">
                            <p class="text-sm font-semibold text-slate-800 dark:text-white truncate">{{ request.user.get_full_name|default:request.user.username }}</p>""")

# 6. Sidebar links
content = content.replace(
"text-slate-300 hover:bg-slate-800 hover:text-white",
"text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-800 hover:text-brand-blue dark:hover:text-white"
)

# 7. Bottom elements border
content = content.replace(
"""            <!-- Bottom elements (Theme toggle & Logout) -->
            <div class="px-2 mt-auto pt-6 border-t border-slate-800 flex flex-col gap-2">""",
"""            <!-- Bottom elements (Theme toggle & Logout) -->
            <div class="px-2 mt-auto pt-6 border-t border-slate-200 dark:border-slate-800 flex flex-col gap-2">""")

# 8. Move theme toggle
toggle_html = """                <button id="theme-toggle"
                    class="flex items-center justify-center w-full p-3 rounded-xl bg-slate-800 text-slate-300 hover:bg-slate-700 transition-all hover:ring-2 ring-brand-blue/20">
                    <span id="theme-toggle-dark-icon" class="hidden">
                        <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path
                                d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z">
                            </path>
                        </svg>
                    </span>
                    <span id="theme-toggle-light-icon" class="hidden">
                        <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"></path>
                        </svg>
                    </span>
                    <span class="ml-2 font-medium">Cambiar Modo</span>
                </button>

"""
content = content.replace(toggle_html, "")

top_bar_search = """            <div class="flex items-center gap-4 ml-auto">
                <div class="hidden md:flex flex-col items-end">"""
top_bar_replace = """            <div class="flex items-center gap-4 ml-auto">
                <button id="theme-toggle" type="button" class="text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 focus:outline-none focus:ring-4 focus:ring-slate-200 dark:focus:ring-slate-700 rounded-lg text-sm p-2.5 transition-colors">
                    <span id="theme-toggle-dark-icon" class="hidden">
                        <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"></path>
                        </svg>
                    </span>
                    <span id="theme-toggle-light-icon" class="hidden">
                        <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"></path>
                        </svg>
                    </span>
                </button>
                <div class="hidden md:flex flex-col items-end">"""
content = content.replace(top_bar_search, top_bar_replace)

# 9. Update JS highlighting logic
content = content.replace(
"link.classList.remove('text-slate-300', 'hover:bg-slate-800', 'hover:text-white');",
"link.classList.remove('text-slate-600', 'dark:text-slate-300', 'hover:bg-slate-200', 'dark:hover:bg-slate-800', 'hover:text-brand-blue', 'dark:hover:text-white');"
)

# 10. Fix logout button hover
content = content.replace(
"hover:text-rose-400 hover:bg-rose-500/10",
"hover:text-rose-500 dark:hover:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-500/10"
)

with open('templates/base.html', 'w') as f:
    f.write(content)
