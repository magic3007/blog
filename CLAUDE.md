# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll static blog built on the **TeXt Theme** (v2.2.6). It's a customizable theme for personal sites, blogs, and documentation with iOS 11-style design elements.

**Key Technologies**:
- **Static Site Generator**: Jekyll (Ruby)
- **Theme**: jekyll-TeXt-theme
- **Frontend**: HTML/CSS/JavaScript, Sass/SCSS, jQuery, Font Awesome
- **Build Tools**: npm scripts, Makefile
- **Containerization**: Docker + Docker Compose
- **Code Quality**: ESLint, Stylelint, Commitlint, Husky

## Common Commands

### Development
```bash
# Local development server
npm run serve           # Standard development
npm run dev             # Custom config (docs/_config.dev.yml)
npm run default         # Default config with watch mode

# Production build
npm run build           # Build for production (JEKYLL_ENV=production)
```

### Docker Development
```bash
# Docker-based development environments
npm run docker-dev:default    # Default Docker setup
npm run docker-dev:dev        # Custom config Docker
npm run docker-prod:serve     # Production Docker
```

### Code Quality
```bash
# JavaScript linting
npm run eslint          # Check JavaScript files
npm run eslint-fix      # Fix JavaScript issues

# CSS/SCSS linting
npm run stylelint       # Check SCSS files
npm run stylelint-fix   # Fix SCSS issues
```

### Makefile Commands
```bash
make jekyll             # Run Jekyll with livereload
```

## Code Architecture

### Directory Structure
```
/
├── _config.yml                    # Main Jekyll configuration
├── _data/                         # YAML data files
│   ├── variables.yml              # Theme variables and CDN config
│   ├── navigation.yml             # Navigation menu
│   ├── locale.yml                 # Internationalization
│   ├── licenses.yml               # License information
│   └── authors.yml                # Author profiles
├── _includes/                     # Template partials
│   ├── scripts/                   # JavaScript components
│   │   ├── components/            # UI components (lightbox, search, etc.)
│   │   ├── lib/                   # Utility libraries
│   │   └── utils/                 # Helper functions
│   ├── search-providers/          # Search implementations
│   └── pageview-providers/        # Analytics integrations
├── _layouts/                      # Page layouts
│   ├── article.html               # Article layout
│   ├── home.html                  # Homepage layout
│   ├── page.html                  # Standard page layout
│   └── 404.html                   # Custom 404 page
├── _posts/                        # Blog posts (Markdown files)
├── _sass/                         # Sass stylesheets
│   ├── custom.scss                # Custom styles (additions here)
│   ├── components/                # Component styles
│   ├── layout/                    # Layout styles
│   └── skins/                     # Theme skins
├── assets/                        # Static assets
│   ├── css/                       # Compiled CSS
│   ├── images/                    # Site images
│   └── img/                       # Article images
├── docker/                        # Docker configurations
│   ├── docker-compose.*.yml       # Various environment configs
│   └── nginx.conf                 # Nginx reverse proxy config
```

### Key Configuration Files
- **`_config.yml`**: Site title, description, author, URL, theme settings
- **`_data/variables.yml`**: Theme customization, CDN URLs, feature toggles
- **`_data/navigation.yml`**: Navigation menu structure
- **`Gemfile`**: Ruby dependencies (gemspec references `jekyll-text-theme.gemspec`)
- **`package.json`**: npm scripts and development dependencies

## Development Standards

### Git Commit Convention
Project uses **Conventional Commits** (configured in `.commitlintrc.js`):
- Format: `type(scope): description` (max 72 characters)
- Types: build, chore, ci, docs, feat, fix, improvement, perf, refactor, release, revert, style, test
- Pre-commit hooks enforce this via Husky

### JavaScript Standards (`.eslintrc`)
- Semicolons required
- Single quotes for strings
- 2-space indentation
- No console statements in production code
- Files in `_includes/**/*.js` are linted

### CSS/SCSS Standards (`.stylelintrc`)
- Based on stylelint-config-standard and stylelint-config-recommended-scss
- Property order by functional groups
- No `!important` declarations
- Double quotes for strings
- No vendor prefixes (use autoprefixer)

## Context7 MCP Integration

**Status**: ✅ Installed and connected (project-specific configuration)

**What it does**: Context7 provides up-to-date, version-specific documentation and code examples for libraries and APIs. It helps avoid outdated information and hallucinated APIs.

**Usage**:
- Add `use context7` to your prompt when you need library/API documentation
- Or specify a library: `use library /supabase/supabase for API and docs`
- To get version-specific docs: `How do I set up Next.js 14 middleware? use context7`

**Example prompts**:
```txt
Create a Next.js middleware that checks for a valid JWT in cookies
and redirects unauthenticated users to `/login`. use context7
```

```txt
Configure a Cloudflare Worker script to cache
JSON API responses for five minutes. use context7
```

**Configuration**: Context7 MCP server is configured to run locally via `npx -y @upstash/context7-mcp`. It's installed without an API key, which means rate limits apply. For higher rate limits, get a free API key from [context7.com/dashboard](https://context7.com/dashboard).

## Custom Features Implemented

Based on Cursor enhancement plans, the following custom features have been added:

1. **Reading Progress Bar** - Top progress bar in articles (`_includes/article/top/custom.html`, `_sass/custom.scss`)
2. **Estimated Reading Time** - Displayed in article metadata (`_includes/article-info.html`)
3. **Code Block Copy Buttons** - One-click copy for code blocks (`_includes/main/bottom/custom.html`)
4. **Terminal-style 404 Page** - Custom 404 with typing animation (`404.html`)
5. **Back-to-top Button** - Floating button appears after scrolling (`_includes/main/bottom/custom.html`)
6. **Open Graph Meta Tags** - Social sharing previews (`_includes/head/custom.html`)
7. **Image Lightbox** - Click-to-enlarge for article images (enabled in `_config.yml` defaults, modified `_includes/scripts/components/lightbox.js`)

## Deployment Options

### Local Development
```bash
npm run serve        # http://localhost:4000/blog/
```

### Docker Deployment
```bash
# Development
npm run docker-dev:default

# Production
npm run docker-prod:serve
```

### GitHub Pages
- Automatically builds from `master` branch
- Uses Jekyll build process

### Traditional Server
```bash
npm run build        # Outputs to `_site/` directory
# Deploy `_site/` contents to any static hosting
```

## Theme Features

The TeXt Theme provides:
- 6 built-in skins: default, dark, forest, ocean, chocolate, orange
- 5 code highlight themes (Tomorrow variants)
- Internationalization support
- Built-in search functionality
- Math formula support (MathJax)
- Diagram support (mermaid, chartjs)
- Third-party comments (Disqus, Gitalk, Valine)
- Page view statistics (LeanCloud)
- Analytics (Google Analytics)
- Social sharing (AddToAny, AddThis)