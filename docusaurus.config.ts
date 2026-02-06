import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  // ====================================================
  // 1. Site Metadata
  // ====================================================
  title: 'LLMOps Docs',
  tagline: 'Scalable AI Infrastructure',
  favicon: 'img/favicon.ico',
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // ====================================================
  // 2. Deployment & Path Configuration
  // ====================================================
  url: 'https://JackSteve-code.github.io', // Updated with your URL
  baseUrl: '/Scalable-llmop/',           // Matches your GitHub repo name
  organizationName: 'JackSteve-code',
  projectName: 'Scalable-llmop',
  trailingSlash: false,
  onBrokenLinks: 'warn',                 // Prevents build failures from missing links

  // ====================================================
  // 3. Presets & Plugins (Content Settings)
  // ====================================================
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/', // Serves docs at the root URL
          editUrl: 'https://github.com/JackSteve-code/Scalable-llmop/tree/main/',
        },
        blog: false, // Disabled to focus purely on documentation
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  // ====================================================
  // 4. Theme Configuration (UI Settings)
  // ====================================================
  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    
    navbar: {
      title: 'LLMOps Docs',
      logo: {
        alt: 'LLMOps Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          href: 'https://github.com/JackSteve-code/Scalable-llmop',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Scalable LLMOps Pipeline',
              to: '/',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/JackSteve-code/Scalable-llmop',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} LLMOps Project. Built with Docusaurus.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;