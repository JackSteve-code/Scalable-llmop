import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'JACK.SYSTEMS', 
  tagline: 'Software and ML Engineer specializing in AI/ML infrastructure',
  favicon: 'img/favicon.ico',

  url: 'https://JackSteve-code.github.io',
  baseUrl: '/Scalable-llmop/',
  organizationName: 'JackSteve-code',
  projectName: 'Scalable-llmop',
  trailingSlash: false,
  onBrokenLinks: 'warn',

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/', 
        },
        blog: false, 
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'JACK.SYSTEMS',
      logo: {
        alt: '',
        src: 'img/logo.svg', 
        style: { display: 'none' }, // STOPS THE LOGO FROM RENDERING
      },
      items: [
        {
          href: 'https://github.com/JackSteve-code/Scalable-llmop',
          label: 'GitHub Source',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'light',
      links: [],
      copyright: `Copyright © ${new Date().getFullYear()} Jack Steve | AI/ML Infrastructure Expert`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
